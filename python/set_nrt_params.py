

def set_nrt_params(b, gpu_id):

    b.update("TELESCOP", "Nancay")
    b.update("FRONTEND", "ROACH")
    b.update("BACKEND", "NUPPI")
    b.update("PKTFMT", "PASP")
    b.update("PKTSIZE", 8208)

    b.update("NRCVR", 2)
    b.update("NBITS", 8)
    b.update("NPOL", 4)
    b.update("POL_TYPE", "IQUV")
    b.update("TRK_MODE", "TRACK")

    # Default params
    b.update("EQUINOX", 2000.0)
    b.update("OBS_MODE", "PSR")
    b.update("OBSFREQ", 1200.0)
    b.update("OBSBW", 512.0)
    b.update("DM", 0.001)
    b.update("CHAN_DM", 0.001)
    b.update("TOTNCHAN", 128)
    b.update("NSUBCHAN", 1)	# Number of subchannels to produce with filterbank
    b.update("N_DS", 4)
    b.update("N_GPU", 2)
    b.update("NBIN", 2048)
    b.update("TFOLD", 60.0)
    b.update("OBS_LEN", 0.0)
    b.update("SCANNUM", 0)
    b.update("SCANLEN", 7200.0)
    b.update("OBSERVER", "GD/IC")
    b.update("SNGPULSE", "False")
    b.update("PROJID", "PSR")

    # ROACH infos
    b.update("DATADIR", "/data/")
    b.update("DATAHOST", "roach")
    if gpu_id == 0:
        b.update("DATAPORT", "6000")
    elif gpu_id == 1:
        b.update("DATAPORT", "6001")
    b.update("PFB_OVER", 8)
    b.update("NBITSADC", 8)

    b.update("CAL_FREQ", 3.335834)
    b.update("CAL_DCYC", 0.5)
    b.update("CAL_PHS", 0.0)
