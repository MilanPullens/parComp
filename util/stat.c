#include <stdio.h>
#include <stdlib.h>
#include <string.h>

int main(int argc, char **argv)
{
    if (argc != 2) {
        printf("Usage: FILENAME\n");
        return EXIT_FAILURE;
    }

    FILE* stream = fopen(argv[1], "r");

    if (stream == NULL) {
        printf("File name not valid.\n");
        return EXIT_FAILURE;
    }

    char line[1024];
    double average = 0.0;
    double smallest = 100000000000000000;
    double largest = -1;
    int nr_of_lines = 0;

    while (fgets(line, 1024, stream))
    {
        double number = strtod(line, NULL);
        if (number < smallest) smallest = number;
        if (number > largest) largest = number;
        average += number;
        nr_of_lines++;
    }

    average /= nr_of_lines;

    printf("%lf,%lf,%lf", average, smallest, largest);

    return EXIT_SUCCESS;
}
