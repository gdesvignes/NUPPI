/* guppi_udp.c
 *
 * UDP implementations.
 */
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <string.h>
#include <sys/types.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include <netdb.h>
#include <fcntl.h>
#include <poll.h>
#include <byteswap.h>
#include <inttypes.h>
#include <errno.h>
#include <netinet/in.h>
#include <arpa/inet.h>

#include "udp.h"
#include "databuf.h"
#include "logging.h"

#define PACKET_SIZE_ORIG ((size_t)8208)
#define PACKET_SIZE_SHORT ((size_t)544)
#define PACKET_SIZE_1SFA ((size_t)8224)
#define PACKET_SIZE_1SFA_OLD ((size_t)8160)
#define PACKET_SIZE_FAST4K ((size_t)4128)
#define PACKET_SIZE_PASP ((size_t)8208)
#define PACKET_SIZE_SONATA ((size_t)4160)

int udp_init(udp_params *p) {

    char strlog[128];

    /* Resolve sender hostname */
    struct addrinfo hints;
    struct addrinfo *result;
    //struct addrinfo *rp;
    memset(&hints, 0, sizeof(struct addrinfo));
    hints.ai_family = AF_INET;
    hints.ai_socktype = SOCK_DGRAM;
    int rv = getaddrinfo(p->sender, NULL, &hints, &result);
    if (rv!=0) {
	log_error("udp_init", "getaddrinfo failed");
	return(ERR_SYS);
    }

    /* Set up socket */
    p->sock = socket(PF_INET, SOCK_DGRAM, IPPROTO_UDP);
    if (p->sock==-1) { 
        log_error("udp_init", "socket error");
        return(ERR_SYS);
    }

    /* bind to local address */
    struct sockaddr_in local_ip;
    local_ip.sin_family =  AF_INET;
    local_ip.sin_port = htons(p->port);
    local_ip.sin_addr.s_addr = INADDR_ANY;
    rv = bind(p->sock, (struct sockaddr *)&local_ip, sizeof(local_ip));
    if (rv==-1) {
        log_error("udp_init", "bind");
        return(ERR_SYS);
    }

    /* Non-blocking recv */
    fcntl(p->sock, F_SETFL, O_NONBLOCK);

    /* Increase recv buffer for this sock */
    int bufsize = 128*1024*1024;
    socklen_t ss = sizeof(int);
    rv = setsockopt(p->sock, SOL_SOCKET, SO_RCVBUF, &bufsize, sizeof(int));
    if (rv<0) { 
        log_error("udp_init", "Error setting rcvbuf size");
    } 
    rv = getsockopt(p->sock, SOL_SOCKET, SO_RCVBUF, &bufsize, &ss); 
    if (0 && rv==0) { 
        sprintf(strlog,"udp_init: SO_RCVBUF=%d\n", bufsize);
        log_info("udp_init",strlog);
    }

    /* Poll command */
    p->pfd.fd = p->sock;
    p->pfd.events = POLLIN;

    return(OK);
}

int udp_forward_init(udp_params *p) {
    char strlog[128];

    /* Set up socket */
    p->sock = socket(PF_INET, SOCK_DGRAM, IPPROTO_UDP);
    if (p->sock==-1) { 
        log_error("udp_init_forward", "socket error");
        return(ERR_SYS);
    }

    /* ensure the socket is reuseable without the painful timeout */
    int on = 1;
    if (setsockopt(p->sock, SOL_SOCKET, SO_REUSEADDR, &on, sizeof(on)) != 0) {
	sprintf(strlog, "setsockopt(SO_REUSEADDR) failed : %s", strerror(errno));
	log_error("udp_init_forward", strlog);
	return(ERR_SYS);
    }	

    /* bind to local address */
    struct sockaddr_in local_ip, dest_ip;
    local_ip.sin_family = AF_INET;
    local_ip.sin_port = htons(p->port+10);;

    dest_ip.sin_family = AF_INET;
    dest_ip.sin_port = htons(p->port+10);  // TODO

    /* Direct packets to one host */
    struct in_addr *addr;
    addr = udp_atoaddr(p->sender);
    if (!addr) {
	sprintf(strlog, "failed atoaddr(%s)", p->sender);
	log_error("udp_init_forward", strlog);
	return(ERR_SYS);
    }
    local_ip.sin_addr.s_addr = addr->s_addr;
    bzero(&(local_ip.sin_zero), 8);

    //addr = udp_atoaddr("192.168.3.12");
    //dest_ip.sin_addr.s_addr = addr->s_addr;

    p->host = local_ip;
    p->sender_addr.ai_addrlen = sizeof(struct sockaddr);

/*
    int rv = bind(p->sock, (struct sockaddr *)&local_ip, sizeof(local_ip));
    if (rv==-1) {
        sprintf(strlog, "bind : %s", strerror(errno));
	log_error("udp_init", strlog);
	return(ERR_SYS);
    }
*/
    return(OK);
}

int udp_wait(udp_params *p) {
    int rv = poll(&p->pfd, 1, 1000); /* Timeout 1sec */
    if (rv==1) { return(OK); } /* Data ready */
    else if (rv==0) { return(TIMEOUT); } /* Timed out */
    else { return(ERR_SYS); }  /* Other error */
}

int udp_recv(udp_params *p, udp_packet *b) {
    int rv = recv(p->sock, b->data, MAX_PACKET_SIZE, 0);
    b->packet_size = rv;
    if (rv==-1) { return(ERR_SYS); }
    else if (p->packet_size) {
        if (rv!=p->packet_size) { return(ERR_PACKET); }
        else { return(OK); }
    } else { 
        p->packet_size = rv;
        return(OK); 
    }
}


int udp_forward(udp_params *p, udp_packet *b) {
    //int rv = send(p->sock, b->data, b->packet_size, 0);
    int rv = sendto(p->sock, b->data, b->packet_size, 0, (struct sockaddr *)&p->host, sizeof(p->host));
    //if (rv==-1) { return(ERR_SYS); }
    if (rv==-1) { fprintf(stderr, "%d %d %s\n", p->host.sin_port, errno, strerror(errno));return(ERR_SYS); }
    else if (rv != b->packet_size) {
	return(ERR_PACKET); 
    } else { 
        return(OK); 
    }
}

struct in_addr *udp_atoaddr(char *address) {
    struct hostent *host;
    static struct in_addr saddr;

    /* First try it as aaa.bbb.ccc.ddd. */
    saddr.s_addr = inet_addr(address);
    if ((int) saddr.s_addr != -1) {
	return &saddr;
    }
    host = gethostbyname(address);
    if (host != NULL) {
	return (struct in_addr *) *host->h_addr_list;
    }
    return NULL;
}


// -- Return the fpga counter --
uint64_t udp_packet_seq_num(const udp_packet *p) {
    /*
     *  8 bytes for PASP, increment by number of samples in the packet !
     */
    if(p->packet_size==PACKET_SIZE_PASP)
	return(bswap_64(*(uint64_t *)(p->data)) / udp_packet_datasize(p->packet_size) );

    /*
     *  4 bytes for SonATA, increment by packet, positionned at +20bytes (see Billy's memo)
     *  Finally, cast the 4 bytes in 8 bytes to use the existing functions
     */
    else if(p->packet_size==PACKET_SIZE_SONATA) {
        return(*(uint64_t *) ((char *)(p->data) + 5*sizeof(uint32_t)));
    }	    

    else return(*(uint64_t *)((char *)(p->data) ) );
}

// -- Return the packet IP id --
uint64_t udp_packet_IP_id(const udp_packet *p) {
   if(p->packet_size==PACKET_SIZE_PASP) return(bswap_64(*(uint64_t *)(p->data + sizeof(uint64_t))) );
}

// -- Return the packet polar number --
int udp_packet_polar(const udp_packet *p) {
    if(p->packet_size==PACKET_SIZE_SONATA) {
        return((int) (*(uint8_t *) ((char *)(p->data) + 10*sizeof(uint8_t))));
    }	
    else return(0);	
}

// -- Return pointer to correct position --
char *udp_packet_data(const udp_packet *p) {

    /*
     *  PASP : skip the first 2  * 64 bits values + point to right position
     */  
    if(p->packet_size==PACKET_SIZE_PASP) 
        return((char *)(p->data) + 2*sizeof(uint64_t));

    /*
     *  SonATA : skip the first 64 bytes of header
     */  
    else if(p->packet_size==PACKET_SIZE_SONATA)
        return((char *)(p->data) + 8*sizeof(uint64_t)); 

    /*
     *  Default  
     */
    return((char *)(p->data) + 2*sizeof(uint64_t)); 
}


// -- Return size of a packet without the header --
size_t udp_packet_datasize(size_t packet_size) {
    if (packet_size==PACKET_SIZE_PASP) 
        return(packet_size - 2*sizeof(uint64_t));
    else if (packet_size==PACKET_SIZE_SONATA) 
        return(packet_size - 8*sizeof(uint64_t));
    return(packet_size - 2*sizeof(uint64_t));
}


/* Copy the data portion of a udp packet to the given output
 *
 */
void udp_packet_data_copy(char *out, const udp_packet *p) {
    memcpy(out, udp_packet_data(p), udp_packet_datasize(p->packet_size));
}


void udp_packet_data_copy_transpose(char *databuf, int nchan, unsigned block_pkt_idx, unsigned packets_per_block, const udp_packet *p) {
    const unsigned chan_per_packet = nchan;
    const size_t bytes_per_sample = 4;
    const unsigned samp_per_packet = udp_packet_datasize(p->packet_size) / bytes_per_sample / chan_per_packet;
    const unsigned samp_per_block = packets_per_block * samp_per_packet;

    char *iptr, *optr;
    unsigned isamp,ichan;
 
    for (ichan=0; ichan<chan_per_packet; ichan++) {
        iptr = udp_packet_data(p) + ichan * bytes_per_sample;
        optr = databuf + bytes_per_sample * (block_pkt_idx*samp_per_packet + ichan*samp_per_block);
        for (isamp=0; isamp<samp_per_packet; isamp++) {
            memcpy(optr, iptr, bytes_per_sample);

	    iptr += nchan * bytes_per_sample;
	    optr += bytes_per_sample;
	}
    }
}


int udp_close(udp_params *p) {
    close(p->sock);
    return(OK);
}    

