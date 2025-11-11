#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#define N 50
#define CHUNKSIZE 5

void A() {
    printf("b");
}

void B() {
    printf(" who is ");
}
void C() {
    printf("ob ");
}


int main (int argc, char *argv[]) {


    #pragma omp parallel sections
    {
        #pragma omp section
        B();

        #pragma omp section
        {
            A();
            C();

        }
        // #pragma omp section
        // C();
    }
    
}