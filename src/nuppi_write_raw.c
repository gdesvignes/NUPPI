/* test_net_thread.c
 *
 * Test run net thread.
 */
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <string.h>
#include <pthread.h>
#include <sys/types.h>
#include <sys/socket.h>
#include <signal.h>
#include <poll.h>
#include <getopt.h>
#include <errno.h>

#include "fitshead.h"
#include "logging.h"
#include "status.h"
#include "databuf.h"
#include "params.h"

#define STATUS "NUPPSTAT"
#include "threads.h"

#include "thread_main.h"

void usage() {
    fprintf(stderr,
            "Usage: nuppi_write_raw [options]\n"
            "Options:\n"
            "  -h, --help        This message\n"
            "  -d, --disk        Write raw data to disk (default no)\n"
            "  -o, --only_net    Run only net_thread\n"
            "  -b, --bands       Select which part of the band to record (0: lowest, 1: highest, 2: both)\n"
           );
}

/* Thread declarations */
void *net_thread(void *_up);
void *rawdisk_thread(void *args);
void *null_thread(void *args);

int main(int argc, char *argv[]) {

    static struct option long_opts[] = {
        {"help",    0, NULL, 'h'},
        {"disk",    0, NULL, 'd'},
        {"only_net",0, NULL, 'o'},
        {"bands",   0, NULL, 'b'},
        {0,0,0,0}
    };
    int opt, opti;
    int disk=0, only_net=0, cbands=0;
    char basename[256];
    while ((opt=getopt_long(argc,argv,"hdob:",long_opts,&opti))!=-1) {
        switch (opt) {
            case 'd':
                disk=1;
                break;
            case 'o':
                only_net=1;
                break;
            case 'b':
				cbands = atoi(optarg);
                break;
            default:
            case 'h':
                usage();
                exit(0);
                break;
        }
    }

    // -- First part of the band --
    thread_args net_args;
    thread_args null_args;
    pthread_t net_thread_id;
    pthread_t disk_thread_id=0;
    status stat;
    if (cbands == 0 || cbands ==2) {

		// -- thread args, start at 1 --
		thread_args_init(&net_args);
		net_args.output_buffer = 1; net_args.gpu_id = 1; net_args.priority = 15;

		// -- Init status shared mem --
		databuf *dbuf=NULL;
		int rv = status_attach(&stat, net_args.gpu_id);
		if (rv!=OK) {
			log_error("nuppi_write_raw", "Error connecting to status shm");
			exit(1);
		}
	   
		// -- Read status shm, init RUN and read filename --
		status_lock_safe(&stat);
		hgets(stat.buf, "BASENAME", 256, basename);
		hputi4(stat.buf, "RUN", 1);
		status_unlock_safe(&stat);


		dbuf = databuf_attach(net_args.output_buffer);
		/* If attach fails, first try to create the databuf */
		if (dbuf==NULL) dbuf = databuf_create(8, 128*1024*1024, net_args.output_buffer);
		/* If that also fails, exit */
		if (dbuf==NULL) {
			log_error("nuppi_write_raw", "Error connecting to databuf net shm");
			exit(1);
		}
		databuf_clear(dbuf);

		/* Launch net thread */
		rv = pthread_create(&net_thread_id, NULL, net_thread, (void *)&net_args);
		if (rv) { 
			log_error("nuppi_write_raw", "Error creating net thread");
			perror("pthread_create");
			exit(1);
		}

		/* Launch raw disk (or null) thread */
		thread_args_init(&null_args);
		null_args.input_buffer = net_args.output_buffer;
		null_args.output_buffer = 2;
		null_args.gpu_id = 1; null_args.priority = 15;

		if (only_net==0) {
			if (disk) rv = pthread_create(&disk_thread_id, NULL, rawdisk_thread, (void *)&null_args);
			else rv = pthread_create(&disk_thread_id, NULL, null_thread, (void *)&null_args);
			if (rv) { 
				log_error("nuppi_write_raw", "Error creating rawdisk/null thread");
				exit(1);
			}
		}
    }


    // -- Use second part of the band --
    thread_args net_args2;
    thread_args null_args2;
    pthread_t net_thread_id2;
    pthread_t disk_thread_id2=0;
    status stat2;
    if(cbands == 1 || cbands ==2) {
		// -- thread args, start at 1 --
		thread_args_init(&net_args2);
		net_args2.output_buffer = 5; net_args2.gpu_id = 2; net_args2.priority = 15;

		// -- Init status shared mem --
		databuf *dbuf=NULL;
		int rv = status_attach(&stat2, net_args2.gpu_id);
		if (rv!=OK) {
			log_error("nuppi_write_raw", "Error connecting to status shm");
			exit(1);
		}
	   
		// -- Read status shm, init RUN and read filename --
		status_lock_safe(&stat2);
		hgets(stat2.buf, "BASENAME", 256, basename);
		hputi4(stat2.buf, "RUN", 1);
		status_unlock_safe(&stat2);


		dbuf = databuf_attach(net_args2.output_buffer);
		/* If attach fails, first try to create the databuf */
		if (dbuf==NULL) dbuf = databuf_create(8, 128*1024*1024, net_args2.output_buffer);
		/* If that also fails, exit */
		if (dbuf==NULL) {
			log_error("nuppi_write_raw", "Error connecting to databuf net shm");
			exit(1);
		}
		databuf_clear(dbuf);

		/* Launch net thread */
		rv = pthread_create(&net_thread_id2, NULL, net_thread, (void *)&net_args2);
		if (rv) { 
			log_error("nuppi_write_raw", "Error creating net thread");
			perror("pthread_create");
			exit(1);
		}

		/* Launch raw disk (or null) thread */
		thread_args_init(&null_args2);
		null_args2.input_buffer = net_args2.output_buffer;
		null_args2.output_buffer = 6; null_args2.gpu_id = 2; null_args2.priority = 15;

		pthread_t disk_thread_id2=0;
		if (only_net==0) {
			if (disk) rv = pthread_create(&disk_thread_id2, NULL, rawdisk_thread, (void *)&null_args2);
			else rv = pthread_create(&disk_thread_id2, NULL, null_thread, (void *)&null_args2);
			if (rv) { 
				log_error("nuppi_write_raw", "Error creating rawdisk/null thread");
				exit(1);
			}
		}

    }

    // -- Run Signal --
    run=1;
    signal(SIGINT, cc);


    /* Wait for end */
    while (run) { 
        sleep(1);
		// Read the RUN keyword in the first status shm, to look for a stop order
	    if(cbands == 0 || cbands ==2) {
		status_lock_safe(&stat);
		hgeti4(stat.buf, "RUN", &run);
		status_unlock_safe(&stat);

		if (run == 0) log_info("nuppi_write_raw", "GPU #0 : Caught RUN = 0 signal for end of observation");
	    }	

	    if(cbands == 1 || cbands ==2) {
		status_lock_safe(&stat2);
		hgeti4(stat2.buf, "RUN", &run);
		status_unlock_safe(&stat2);
		
		if (run == 0) log_info("nuppi_write_raw", "GPU #1 : Caught RUN = 0 signal for end of observation");
	    }	

		//if (null_args.finished || null_args2.finished) run=0;
    }


    if(cbands == 0 || cbands ==2) {
	    // -- First cancel threads -- 
		if (disk_thread_id) pthread_cancel(disk_thread_id);
		pthread_cancel(net_thread_id);

	    // -- Then kill threads -- 
		if (disk_thread_id) pthread_kill(disk_thread_id,SIGINT);
		pthread_kill(net_thread_id,SIGINT);

		// -- Finally join --
		if (disk_thread_id) {
			pthread_join(disk_thread_id,NULL);
			log_info("nuppi_write_raw", "Joined disk thread");
		}
		pthread_join(net_thread_id,NULL);
		log_info("nuppi_write_raw", "Joined net thread");

		// -- Destroy args --
		thread_args_destroy(&net_args);
		thread_args_destroy(&null_args);
	}	

    if(cbands == 1 || cbands ==2) {
	    // -- First cancel threads -- 
		if (disk_thread_id2) pthread_cancel(disk_thread_id2);
		pthread_cancel(net_thread_id2);

	    // -- Then kill threads -- 
		if (disk_thread_id2) pthread_kill(disk_thread_id2,SIGINT);
		pthread_kill(net_thread_id2,SIGINT);

		// -- Finally join --
		if (disk_thread_id2) {
			pthread_join(disk_thread_id2,NULL);
			log_info("nuppi_write_raw", "Joined disk thread2");
		}
		pthread_join(net_thread_id2,NULL);
		log_info("nuppi_write_raw", "Joined net thread2");

		// -- Destroy args --
		thread_args_destroy(&net_args2);
		thread_args_destroy(&null_args2);
	}	

    // -- Log file --
    char filename[128], hostname[128];
    gethostname(hostname, 127);
    sprintf(filename, "/home/pulsar/data/%s-%s.log", basename, hostname);
    rename(LOG_FILENAME, filename);
    char strlog[128];
    sprintf(strlog, "Moving log file to %s", filename);
    log_info("nuppi_daq_ds", strlog);


    exit(0);
}
