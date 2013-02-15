#ifndef __STRUCTURE__
#define __STRUCTURE__
#include<vector>
struct HostSources{
    std::vector<int> x_source_position;
    std::vector<int> y_source_position;
    std::vector<int> source_type;
    //Variance acts as amplitude in the case of constant
    //and sinusoid source;
    //Mean acts as the frequency in the case of sinusoid sources.
    std::vector<float> mean;
    std::vector<float> variance;

    int get_size(){
        return x_source_position.size();
    }
};
#endif
