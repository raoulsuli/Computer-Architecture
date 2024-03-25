#include <math.h>
#include <stddef.h>
#include <stdio.h>
#include <stdint.h>     // provides int8_t, uint8_t, int16_t etc.
#include <stdlib.h>

struct particle
{
    int8_t v_x, v_y, v_z;
};

int main(int argc, char* argv[])
{
    if(argc < 2)
    {
        printf("apelati cu %s <n>\n", argv[0]);
        return -1;
    }

    long n = atol(argv[1]);

    struct particle *vect = calloc(sizeof(struct particle), n * n);
    if (!vect) {
        perror("malloc failed");
        exit(EXIT_FAILURE);
    }

    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            if (i % 2 == 0) {
                vect[i * n + j].v_x = (int8_t)rand() % 128;
                vect[i * n + j].v_y = (int8_t)rand() % 128;
                vect[i * n + j].v_z = (int8_t)rand() % 128;
            } else {
                vect[i * n + j].v_x = (int8_t)(rand() % 129) * -1;
                vect[i * n + j].v_y = (int8_t)(rand() % 129) * -1;
                vect[i * n + j].v_z = (int8_t)(rand() % 129) * -1;
            }
        }
    }

    int8_t *result = (int8_t *)vect;

    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            result[i * n + j] *= 0.5f;
        }
    }
    // TODO
    // scalati vitezele tuturor particulelor cu 0.5
    //   -> folositi un cast la int8_t* pentru a parcurge vitezele fara
    //      a fi nevoie sa accesati individual componentele v_x, v_y, si v_z

    // compute max particle speed
    float max_speed = 0.0f;
    for(long i = 0; i < n * n; ++i)
    {
        float speed = sqrt(vect[i].v_x * vect[i].v_x +
                           vect[i].v_y * vect[i].v_y +
                           vect[i].v_z * vect[i].v_z);
        if(max_speed < speed) max_speed = speed;
    }

    // print result
    printf("viteza maxima este: %f\n", max_speed);

    free(vect);

    return 0;
}

