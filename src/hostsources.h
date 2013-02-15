#ifndef __STRUCTURE__
#define __STRUCTURE__

#define CONSTANT_SOURCE 0
#define SINUSOID_SOURCE 1
#define GAUSSIAN_SOURCE 2

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

    void add_source(int x, int y, int type, float mean_val, float variance_val){
        x_source_position.push_back(x);
        y_source_position.push_back(y);
        source_type.push_back(type);
        mean.push_back(mean_val);
        variance.push_back(variance_val);
    }
};
#endif
