/*! @file datablock.h
 * @author Bharath M R
 * @brief Contains all the information about the simulation.
 *
 * This is the structure which contains all the arrays required
 * for simulation.
 */
#ifndef __DATABLOCK__
#define __DATABLOCK__

#include<vector>
#include "structure.h"
#include "devicesources.h"
#include "hostsources.h"
#include "cpu_anim.h"
#include <string>
using namespace std;
// This gives the best results
#define BLOCKSIZE_X 256 //! Blocksizes used for simulation
#define BLOCKSIZE_Y 1


#define BLOCKSIZE_HX 256 //! Block size used for hx updates.
#define BLOCKSIZE_HY 1

#define OUTPUT_ANIM 0   //! Output animation or hdf5 files
#define OUTPUT_HDF5 1

// Different simulation types.
// TODO ? (Not sure) Replace it with an enumerate?
#define TM_SIMULATION 0
#define TM_PML_SIMULATION 1
#define DRUDE_SIMULATION 2

//! TM Fields array indices.
#define TM_HXFIELD 0
#define TM_HYFIELD 1
#define TM_EZFIELD 2

//! Constant indices
#define MUINDEX 0
#define EPSINDEX 1
#define SIGMAINDEX 2
#define SIGMA_STAR_INDEX 3

//! TM PML array indices
#define TM_PML_HXFIELD 0
#define TM_PML_HYFIELD 1
#define TM_PML_EZXFIELD 2
#define TM_PML_EZYFIELD 3
#define TM_PML_EZFIELD 4

//! Constant Indices for PML TM simulation
#define SIGMAINDEX_X 2
#define SIGMA_STAR_INDEX_X 3
#define SIGMAINDEX_Y 4
#define SIGMA_STAR_INDEX_Y 5

//! Drude Mode Indices
#define DRUDE_HXFIELD 0
#define DRUDE_HYFIELD 1
#define DRUDE_EZFIELD 2
#define DRUDE_JFIELD 3 // want of a better name
#define DRUDE_EZOLD 4

//! Drude Mode constants
#define GAMMA_INDEX 4
#define OMEGAP_INDEX 5

struct Datablock{
    unsigned char *output_bitmap; //! bitmap file to copy into
    float ** fields; //! Device pointer to different field arrays
    float ** constants; //! Device pointer to constant arrays.
    float *save_field; //! Host pointer to the field that has to be copied into.
    float ** coefs; //! Device pointer to all the coefficients.
    CPUAnimBitmap *bitmap; //! Host bitmap structure pointer.
    cudaEvent_t start, stop; //! for profiling. ignore.
    float totalTime; //! For profiling. Keeps track of time for a certain number of simulations.
    int present_ticks;
    float frames; //! Profiling.
    int simulationType; //! Simulation type: TM mode. TM PML, Drude
    int outputType;
    Structure * structure; //! Pointer to the structure array(Contains xdim, ydim)
    int number_of_sources;
    DeviceSources * sources; //! Device pointer to the structure of arrays which contains the source definitions
    string simulation_name;

    Datablock(int type, int output_type){
        simulationType = type;
        outputType = output_type;
        if(type == TM_SIMULATION){
            fields = (float **) malloc(sizeof(float *) * 3);
            constants = (float **) malloc(sizeof(float *) * 4);
            coefs = (float **) malloc(sizeof(float *) * 4);
        }

        else if(type == TM_PML_SIMULATION){
            fields = (float **) malloc(sizeof(float *) * 5);
            constants = (float **) malloc(sizeof(float *) * 6);
            coefs = (float **) malloc(sizeof(float *) * 8);
        }

        else if(type == DRUDE_SIMULATION){
            fields = (float **) malloc(sizeof(float *) * 5);
            constants = (float **) malloc(sizeof(float *) * 6);
            coefs = (float **) malloc(sizeof(float *) * 7);
        }
    }

    /*! Destructor which frees all the array pointers allocated */
    ~Datablock(){
        free(fields);
        free(constants);
        free(coefs);
    }

};
#endif
