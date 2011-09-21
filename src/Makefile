CUDA = /opt/cuda
CFLAGS =  -O3 -mtune=native -march=native -funroll-loops -Wall -I$(CUDA)/include -I/usr/include/libcfitsio0
NVCCFLAGS = -arch sm_13 -I$(CUDA)/include -I/usr/include/libcfitsio0
INSTALL_DIR = ../bin

#PROGS = check_databuf check_status clean_shmem \
	test_udp_recv test_psrfits test_psrfits_read fold_psrfits \
	fix_psrfits_polyco psrfits_singlepulse
PROGS = 
OBJS  = status.o databuf.o udp.o logging.o \
	params.o mjdtime.o thread_args.o \
	write_psrfits.o read_psrfits.o misc_utils.o \
	fold.o polyco.o hget.o hput.o sla.o downsample.o \
	dedisperse_utils.o cpu_utils.o

THREAD_PROGS = nuppi_daq_ds nuppi_daq_dedisp nuppi_write_raw 
THREAD_OBJS  = net_thread.o rawdisk_thread.o psrfits_thread.o fold_thread.o null_thread.o

CUDA_OBJS = dedisp_thread.o dedisp_ds_thread.o ds_thread.o fold_gpu.o dedisperse_gpu.o downsample_gpu.o
LIBS = -lcfitsio -L$(PRESTO)/lib -lsla -lm -lpthread
CUDA_LIBS = -L$(CUDA)/lib64 -lcufft -lcuda -lcudart -lrt -lm

all: $(PROGS) $(THREAD_PROGS) 
clean:
	rm -f $(PROGS) $(THREAD_PROGS) psrfits.tgz *~ *.o test_psrfits_0*.fits *.ptx

install: $(PROGS) $(THREAD_PROGS) 
	mkdir -p $(INSTALL_DIR) && \
	cp -f $(PROGS) $(THREAD_PROGS) psrfits_subband test_dedisp_thread nuppi_daq_dedisp $(INSTALL_DIR)

psrfits.tgz: psrfits.h read_psrfits.c write_psrfits.c polyco.c polyco.h \
	PSRFITS_v3.4_fold_template.txt \
	PSRFITS_v3.4_search_template.txt
	tar cvzf $@ $^

find_dropped_blocks: find_dropped_blocks.o 
	$(CC) $(CFLAGS) $< -o $@ -lcfitsio -lm

psrfits_subband: psrfits_subband.c psrfits_subband_cmd.o $(OBJS)
	$(CC) $(CFLAGS) $< -o $@ psrfits_subband_cmd.o $(OBJS) $(LIBS)

%.o : %.cu
	nvcc -c $(NVCCFLAGS) $< -o $@

test_dedisp_thread: test_dedisp_thread.c $(THREAD_OBJS) $(OBJS) $(CUDA_OBJS)
	$(CC) $(CFLAGS) $(CUDA_CFLAGS) $< -o $@ $(THREAD_OBJS) \
		$(CUDA_OBJS) $(OBJS) $(LIBS) $(CUDA_LIBS)

nuppi_daq_dedisp: nuppi_daq_dedisp.c $(THREAD_OBJS) $(OBJS) $(CUDA_OBJS)
	$(CC) $(CFLAGS) $(CUDA_CFLAGS) $< -o $@ $(THREAD_OBJS) \
		$(CUDA_OBJS) $(OBJS) $(LIBS) $(CUDA_LIBS)

nuppi_daq_ds: nuppi_daq_ds.c $(THREAD_OBJS) $(OBJS) $(CUDA_OBJS)
	$(CC) $(CFLAGS) $(CUDA_CFLAGS) $< -o $@ $(THREAD_OBJS) \
		$(CUDA_OBJS) $(OBJS) $(LIBS) $(CUDA_LIBS)

nuppi_write_raw: nuppi_write_raw.c $(THREAD_OBJS) $(OBJS) $(CUDA_OBJS)
	$(CC) $(CFLAGS) $(CUDA_CFLAGS) $< -o $@ $(THREAD_OBJS) \
		$(CUDA_OBJS) $(OBJS) $(LIBS) $(CUDA_LIBS)

.SECONDEXPANSION:
$(PROGS): $$@.c $(OBJS)
	$(CC) $(CFLAGS) $< -o $@ $(OBJS) $(LIBS) $(THREAD_LIBS)
