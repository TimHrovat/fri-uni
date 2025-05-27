#include "main.h"

#define RCC_AHB4ENR ((volatile uint32_t *)0x580244E0)

#define GPIOA_CLK_ENABLE() (*RCC_AHB4ENR |= (1 << 0))
#define GPIOB_CLK_ENABLE() (*RCC_AHB4ENR |= (1 << 1))
#define GPIOC_CLK_ENABLE() (*RCC_AHB4ENR |= (1 << 2))
#define GPIOD_CLK_ENABLE() (*RCC_AHB4ENR |= (1 << 3))
#define GPIOE_CLK_ENABLE() (*RCC_AHB4ENR |= (1 << 4))
#define GPIOF_CLK_ENABLE() (*RCC_AHB4ENR |= (1 << 5))
#define GPIOG_CLK_ENABLE() (*RCC_AHB4ENR |= (1 << 6))
#define GPIOH_CLK_ENABLE() (*RCC_AHB4ENR |= (1 << 7))
#define GPIOI_CLK_ENABLE() (*RCC_AHB4ENR |= (1 << 8))
#define GPIOJ_CLK_ENABLE() (*RCC_AHB4ENR |= (1 << 9))
#define GPIOK_CLK_ENABLE() (*RCC_AHB4ENR |= (1 << 10))

typedef struct {
	volatile uint32_t MODER;
	volatile uint32_t OTYPER;
	volatile uint32_t OSPEEDR;
	volatile uint32_t PUPDR;
	volatile uint32_t IDR;
	volatile uint32_t ODR;
	volatile uint32_t BSRR;
} GPIO_device;

#define GPIOC ((GPIO_device *)0x58020800)
#define GPIOD ((GPIO_device *)0x58020C00)
#define GPIOI ((GPIO_device *)0x58022000)
#define GPIOJ ((GPIO_device *)0x58022400)

#define GPIO_MODE_INPUT 0
#define GPIO_MODE_OUTPUT 1

#define GPIO_OUTPUT_TYPE_PUSH_PULL 0
#define GPIO_OUTPUT_TYPE_OPEN_DRAIN 1

#define GPIO_NO_PULL 0
#define GPIO_PULL_UP 1
#define GPIO_PULL_DOWN 1

#define GPIO_SPEED_LOW 0
#define GPIO_SPEED_MEDIUM 1
#define GPIO_SPEED_HIGH 2
#define GPIO_SPEED_VERY_HIGH 3

void GPIO_Init(GPIO_device *gpio, uint32_t pin, uint32_t mode, uint32_t otype, uint32_t speed, uint32_t pupd);
void GPIO_WritePin(GPIO_device *gpio, uint32_t pin, uint32_t value);
uint32_t GPIO_ReadPin(GPIO_device *gpio, uint32_t pin);

void delay(uint32_t delay) {
	for (int i = 0; i < delay; i++) {}
}

int main(void) {
	GPIOC_CLK_ENABLE();
	GPIOD_CLK_ENABLE();
	GPIOI_CLK_ENABLE();
	GPIOJ_CLK_ENABLE();

	GPIO_Init(GPIOD, 3, GPIO_MODE_OUTPUT, GPIO_OUTPUT_TYPE_PUSH_PULL, GPIO_SPEED_LOW, GPIO_NO_PULL);
	GPIO_Init(GPIOI, 13, GPIO_MODE_OUTPUT, GPIO_OUTPUT_TYPE_PUSH_PULL, GPIO_SPEED_LOW, GPIO_NO_PULL);
	GPIO_Init(GPIOJ, 2, GPIO_MODE_OUTPUT, GPIO_OUTPUT_TYPE_OPEN_DRAIN, GPIO_SPEED_LOW, GPIO_PULL_UP);
	GPIO_Init(GPIOC, 13, GPIO_MODE_INPUT, GPIO_OUTPUT_TYPE_PUSH_PULL, GPIO_SPEED_LOW, GPIO_NO_PULL);

	while(1) {
		if (GPIO_ReadPin(GPIOC, 13)) {
			GPIO_WritePin(GPIOD, 3, 1);

			delay(1000000);

			GPIO_WritePin(GPIOD, 3, 0);
			GPIO_WritePin(GPIOJ, 2, 0);

			delay(1000000);

			GPIO_WritePin(GPIOJ, 2, 1);
			GPIO_WritePin(GPIOI, 13, 0);

			delay(1000000);

			GPIO_WritePin(GPIOI, 13, 1);
			GPIO_WritePin(GPIOJ, 2, 0);

			delay(1000000);

			GPIO_WritePin(GPIOJ, 2, 1);
			GPIO_WritePin(GPIOD, 3, 1);

			delay(1000000);

			GPIO_WritePin(GPIOD, 3, 0);
		} else {
			GPIO_WritePin(GPIOD, 3, 0);
			GPIO_WritePin(GPIOI, 13, 1);
			GPIO_WritePin(GPIOJ, 2, 1);
		}
	}
}

void GPIO_Init(GPIO_device *gpio, uint32_t pin, uint32_t mode, uint32_t otype, uint32_t speed, uint32_t pupd) {
    gpio->MODER = gpio->MODER & ~(3 << (2 * pin));
    gpio->MODER = gpio->MODER | (mode << (2 * pin));

    gpio->OTYPER = gpio->OTYPER & ~(1 << pin);
    gpio->OTYPER = gpio->OTYPER | (otype << pin);

    gpio->OSPEEDR = gpio->OSPEEDR & ~(3 << (2 * pin));
    gpio->OSPEEDR = gpio->OSPEEDR | (speed << (2 * pin));

    gpio->PUPDR = gpio->PUPDR & ~(3 << (2 * pin));
    gpio->PUPDR = gpio->PUPDR | (pupd << (2 * pin));
}

void GPIO_WritePin(GPIO_device *gpio, uint32_t pin, uint32_t value) {
    if (value == 1) {
      gpio->BSRR = 1 << pin;
    } else {
      gpio->BSRR = 1 << (pin + 16);
    }
}

uint32_t GPIO_ReadPin(GPIO_device *gpio, uint32_t pin) {
    if (gpio->IDR & (1 << pin)) {
      return 1;
    } else {
      return 0;
    }
}
