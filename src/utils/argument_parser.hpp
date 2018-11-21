#ifndef __ARGUMENT_PARSER_H__
#define __ARGUMENT_PARSER_H__

#include "utils.hpp"
#include "parameters.hpp"


void parseArgument(int argc, char* argv[])
{
    for (int i = 1; i < argc; ++ i) {
        if (!strcmp(argv[i], "--input")) {
            fromString(argv[++ i], INPUT_IMAGE);
        } else if (!strcmp(argv[i], "--output")) {
            fromString(argv[++ i], OUTPUT_IMAGE);
        } else if (!strcmp(argv[i], "--model")) {
            fromString(argv[++ i], PRETRAINED_MODEL);
        } else {
            fprintf(stderr, "[Warning] Unknown Parameter: %s\n", argv[i]);
        }
    }
}

#endif