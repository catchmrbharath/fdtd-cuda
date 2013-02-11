#ifndef _STRUCTURE_H_
#define _STRUCTURE_H_
#include "cuda.h"
#include<vector>


class structure{
    public:
        int xdim; //No. of x values in the grid
        int ydim; //No of y values in the grid.
        float xsize; //length of the actual physical structure.
        float ysize;
        float delta;
        float deltat;
        float S;
        float* sigma;
        float* epsilon;
        float* mu;
        float* sigmastar;
        void calculate_dim();
        void calculate_deltat();
        void set_background_epsilonr(float epsilonr);
        void set_background_mur(float mur);
        void set_background_sigma(float sigma);
        void set_background_sigma_star(float sigma_star);
        void populate_epsilon(const vector<pair<int, int> > coords);
        void populate_mu(const vector< pair<int, int> > coords);
        void populate_sigma(const vector<pair<int, int> > coords);
        void populate_sigma_star(const vector<pair<int, int> > coords);
        void read_structure(char * filename);

} Structure;
