#include "debug.h"
#include "structure.h"
#include<istringstream>

void Structure::read_structure(char * filename){
    FILE *file = fopen(filename, 'r');
    string temp;
    while(
