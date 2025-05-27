#include "main.h"
#include <stdint.h>

#define RCC_AHB4ENR ((volatile uint32_t *)0x580244E0)

// led 1
#define GPIOD_MODER ((volatile uint32_t *)0x58020C00)
#define GPIOD_ODR ((volatile uint32_t *)0x58020C14)

// led 2
#define GPIOJ_MODER ((volatile uint32_t *)0x58022400)
#define GPIOJ_ODR ((volatile uint32_t *)0x58022414)

// led 3
#define GPIOI_MODER ((volatile uint32_t *)0x58022000)
#define GPIOI_ODR ((volatile uint32_t *)0x58022014)

// blue btn
#define GPIOC_MODER ((volatile uint32_t *) 0x58020800)
#define GPIOC_IDR ((volatile uint32_t *) 0x58020810)

void delay(uint32_t delay) {
	for (int i = 0; i < delay; i++) {}
}

int main(void)
{
    // vklopimo uro za GPIOD, GPIOC, GPIOI in GPIOJ
    *RCC_AHB4ENR = *RCC_AHB4ENR | (1 << 3) | (1 << 2) | (1 << 8) | (1 << 9);

    // init LED 1 (GPIOD pin 3)
    *GPIOD_MODER = *GPIOD_MODER & ~(3 << (2 * 3));
    *GPIOD_MODER = *GPIOD_MODER | (1 << (2 * 3));

    // init LED 2 (GPIOJ pin 2)
    *GPIOJ_MODER = *GPIOJ_MODER & ~(3 << (2 * 2));
    *GPIOJ_MODER = *GPIOJ_MODER | (1 << (2 * 2));

    // init LED 3 (GPIOI pin 13)
    *GPIOI_MODER = *GPIOI_MODER & ~(3 << (2 * 13));
    *GPIOI_MODER = *GPIOI_MODER | (1 << (2 * 13));

    // init gumb
    *GPIOC_MODER = *GPIOC_MODER & ~(3 << (2 * 13));

    while(1) {
        uint32_t stanje_gumba = *GPIOC_IDR & (1 << 13);

        if (stanje_gumba) {
            // prižgemo LED 1, ugasnemo LED 2 in 3
            *GPIOD_ODR = *GPIOD_ODR | (1 << 3);    // LED 1 ON

            delay(1000000);
            *GPIOD_ODR = *GPIOD_ODR & ~(1 << 3);   // LED 1 OFF
            *GPIOJ_ODR = *GPIOJ_ODR & ~(1 << 2);   // LED 2 ON

            delay(1000000);
            *GPIOJ_ODR = *GPIOJ_ODR | (1 << 2);    // LED 2 OFF
            *GPIOI_ODR = *GPIOI_ODR & ~(1 << 13);  // LED 3 ON

            delay(1000000);
            *GPIOI_ODR = *GPIOI_ODR | (1 << 13);   // LED 3 OFF
            *GPIOJ_ODR = *GPIOJ_ODR & ~(1 << 2);   // LED 2 ON

            delay(1000000);
            *GPIOJ_ODR = *GPIOJ_ODR | (1 << 2);    // LED 2 OFF
            *GPIOD_ODR = *GPIOD_ODR | (1 << 3);    // LED 1 ON

            delay(1000000);
            *GPIOD_ODR = *GPIOD_ODR & ~(1 << 3);   // LED 1 OFF
        } else {
            *GPIOD_ODR = *GPIOD_ODR & ~(1 << 3);   // LED 1 OFF
            *GPIOJ_ODR = *GPIOJ_ODR | (1 << 2);    // LED 2 OFF
            *GPIOI_ODR = *GPIOI_ODR | (1 << 13);   // LED 3 OFF
        }
    }
}
