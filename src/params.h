/* params.h 
 *
 * Defines structure used internally to represent observation 
 * parameters.  Includes routines to read/write this info to
 * a "FITS-style" shared memory buffer.
 */
#ifndef _PARAMS_H
#define _PARAMS_H

#include "psrfits.h"
#include "udp.h"

typedef struct {
    /* Packet information for the current block */
    long long packetindex;      // Index of first packet in raw data block
    double drop_frac_avg;       // Running average of the fract of dropped packets
    double drop_frac_tot;       // Total fraction of dropped packets
    double drop_frac;           // Fraction of dropped packets in this block
    int packetsize;             // Size in bytes of data portion of each packet
    int n_packets;              // Total number of packets in current block
    int n_dropped;              // Number of packets dropped in current block
    int packets_per_block;      // Total number of packets per block
    int stt_valid;              // Has an accurate start time been measured
    /* Backend hardware info */
    int decimation_factor;      // Number of raw spectra integrated
    int n_bits_adc;             // Number of bits sampled by ADCs
    int pfb_overlap;            // PFB overlap factor
    float scale[16*1024];       // Per-channel scale factor
    float offset[16*1024];      // Per-channel offset
/* GD GD
    int n_bits;                 // Number of bits for a sample
    int n_pol;                  // Number of polarization recorded
    int n_gpu;                  // Number of GPUs per DS
    int n_samp;                 // Number of samples per packet
    int n_bloc;                 // Number of databuf block
    int blocsize;               // Size of a databuf block
    int nchan;                  // Number of channels per DS
    int obsnchan;               // Number of channels per DS
*/
} obs_params;

void read_obs_mode(char *buf, char *mode);
int read_net_params(char *buf, udp_params *u);
int read_fwd_params(char *buf, udp_params *u);
int read_status_params(char *buf, obs_params *g, struct psrfits *p);
void read_subint_params(char *buf, obs_params *g, struct psrfits *p);
void free_psrfits(struct psrfits *p);
#endif
