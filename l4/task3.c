#include <stdio.h>
#include <stdint.h>     // provides int8_t, uint8_t, int16_t etc.
#include <stdlib.h>

struct a
{
    int64_t x;
    int32_t y, z;
};

struct b
{
    int32_t x;
    int64_t y;
    int32_t z;
};

int main(void)
{
    int32_t i1;
    int64_t l1;
    struct a a1;
    struct b b1;
    int32_t i2, i3, i4, i5;
    int64_t l2, l3, l4, l5;
    struct a a2, a3, a4, a5;
    struct b b2, b3, b4, b5;

    printf("%p\n%p\n%p\n%p\n", &i1, &l1, &a1, &b1);
    printf("%p\n%p\n%p\n%p\n", &i2, &l2, &a2, &b2);
    printf("%p\n%p\n%p\n%p\n", &i3, &l3, &a3, &b3);
    printf("%p\n%p\n%p\n%p\n", &i4, &l4, &a4, &b4);
    printf("%p\n%p\n%p\n%p\n", &i5, &l5, &a5, &b5);

    printf("%zu\n%zu\n", sizeof(struct a), sizeof(struct b));

    void* aux_vect = malloc(10 * sizeof(float) + 31);
    float* vect = (float*)(((uintptr_t)aux_vect + 31) & ~31);

    printf("aux_vect: %p\n", aux_vect);
    printf("vect:     %p\n", vect);

    // trebuie se eliberam memoria folosind adresa intiala, returnata de malloc
    free(aux_vect);

    return 0;
}

