/* USER CODE BEGIN Header */
/**
  ******************************************************************************
  * @file           : main.c
  * @brief          : Main program body
  ******************************************************************************
  * @attention
  *
  * Copyright (c) 2025 STMicroelectronics.
  * All rights reserved.
  *
  * This software is licensed under terms that can be found in the LICENSE file
  * in the root directory of this software component.
  * If no LICENSE file comes with this software, it is provided AS-IS.
  *
  ******************************************************************************
  */
/* USER CODE END Header */
/* Includes ------------------------------------------------------------------*/
#include "main.h"

/* Private includes ----------------------------------------------------------*/
/* USER CODE BEGIN Includes */

/* USER CODE END Includes */

/* Private typedef -----------------------------------------------------------*/
/* USER CODE BEGIN PTD */

/* USER CODE END PTD */

/* Private define ------------------------------------------------------------*/
/* USER CODE BEGIN PD */

/* USER CODE END PD */

/* Private macro -------------------------------------------------------------*/
/* USER CODE BEGIN PM */

/* USER CODE END PM */

/* Private variables ---------------------------------------------------------*/

/* USER CODE BEGIN PV */

/* USER CODE END PV */

/* Private function prototypes -----------------------------------------------*/
void SystemClock_Config(void);
static void MPU_Config(void);
/* USER CODE BEGIN PFP */

/* USER CODE END PFP */

/* Private user code ---------------------------------------------------------*/
#define SIZE 40000
uint8_t data[SIZE];
DMA_HandleTypeDef dma1_struct = {0};
UART_HandleTypeDef uart;

void start_timer() {
  __HAL_RCC_TIM2_CLK_ENABLE();
  TIM_HandleTypeDef timer = {0};
  timer.Instance = TIM2;
  timer.Init.CounterMode = TIM_COUNTERMODE_UP;
  timer.Init.Period = 100000000;
  timer.Init.Prescaler = 64 -1;
  HAL_TIM_Base_Init(&timer);

  HAL_TIM_Base_Start(&timer);
  __HAL_TIM_SetCounter(&timer, 0);
}

void stop_timer() {
  TIM_HandleTypeDef timer = {0};
  timer.Instance = TIM2;
  HAL_TIM_Base_Stop(&timer);
}

uint32_t get_time() {
  TIM_HandleTypeDef timer = {0};
  timer.Instance = TIM2;
  return __HAL_TIM_GetCounter(&timer);
}

void init_leds() {
	  // green led
	  GPIO_InitTypeDef led_pd3 = {0};
	  led_pd3.Pin = GPIO_PIN_3;
	  led_pd3.Mode = GPIO_MODE_OUTPUT_PP;
	  led_pd3.Pull = GPIO_NOPULL;
	  led_pd3.Speed = GPIO_SPEED_FREQ_LOW;
	  HAL_GPIO_Init(GPIOD, &led_pd3);

	  // green led
	  GPIO_InitTypeDef led_pj2 = {0};
	  led_pj2.Pin = GPIO_PIN_2;
	  led_pj2.Mode = GPIO_MODE_OUTPUT_PP;
	  led_pj2.Pull = GPIO_NOPULL;
	  led_pj2.Speed = GPIO_SPEED_FREQ_LOW;
	  HAL_GPIO_Init(GPIOJ, &led_pj2);

	  // red led
	  GPIO_InitTypeDef led_pi13 = {0};
	  led_pi13.Pin = GPIO_PIN_13;
	  led_pi13.Mode = GPIO_MODE_OUTPUT_OD;
	  led_pi13.Pull = GPIO_PULLUP;
	  led_pi13.Speed = GPIO_SPEED_FREQ_LOW;
	  HAL_GPIO_Init(GPIOI, &led_pi13);

	  HAL_GPIO_WritePin(GPIOD, GPIO_PIN_3, GPIO_PIN_RESET);
	  HAL_GPIO_WritePin(GPIOJ, GPIO_PIN_2, GPIO_PIN_SET);
	  HAL_GPIO_WritePin(GPIOI, GPIO_PIN_13, GPIO_PIN_SET);
}

void toggle_leds() {
  HAL_GPIO_TogglePin(GPIOD, GPIO_PIN_3);
  HAL_GPIO_TogglePin(GPIOJ, GPIO_PIN_2);
  HAL_GPIO_TogglePin(GPIOI, GPIO_PIN_13);
  HAL_Delay(100);
}


/**
  * @brief  The application entry point.
  * @retval int
  */
int main(void)
{
    MPU_Config();
    HAL_Init();
    SystemClock_Config();

    __HAL_RCC_DMA1_CLK_ENABLE();
    __HAL_RCC_GPIOD_CLK_ENABLE();
    __HAL_RCC_GPIOJ_CLK_ENABLE();
    __HAL_RCC_GPIOI_CLK_ENABLE();
    __HAL_RCC_GPIOB_CLK_ENABLE();
    __HAL_RCC_USART3_CLK_ENABLE();

    init_leds();

    // UART and DMA initialization
    uart.Instance = USART3;
    uart.Init.BaudRate = 115200;
    uart.Init.WordLength = UART_WORDLENGTH_8B;
    uart.Init.StopBits = UART_STOPBITS_1;
    uart.Init.Parity = UART_PARITY_NONE;
    uart.Init.Mode = UART_MODE_TX_RX;
    uart.Init.HwFlowCtl = UART_HWCONTROL_NONE;
    HAL_UART_Init(&uart);

    dma1_struct.Instance = DMA1_Stream0;
    dma1_struct.Init.Request = DMA_REQUEST_USART3_TX;
    dma1_struct.Init.Direction = DMA_MEMORY_TO_PERIPH;
    dma1_struct.Init.PeriphInc = DMA_PINC_DISABLE;
    dma1_struct.Init.MemInc = DMA_MINC_ENABLE;
    dma1_struct.Init.PeriphDataAlignment = DMA_MDATAALIGN_BYTE;
    dma1_struct.Init.MemDataAlignment = DMA_MDATAALIGN_BYTE;
    dma1_struct.Init.Mode = DMA_NORMAL;
    dma1_struct.Init.Priority = DMA_PRIORITY_LOW;
    dma1_struct.Init.FIFOMode = DMA_FIFOMODE_ENABLE;
    dma1_struct.Init.FIFOThreshold = DMA_FIFO_THRESHOLD_1QUARTERFULL;
    HAL_DMA_Init(&dma1_struct);

    __HAL_LINKDMA(&uart, hdmatx, dma1_struct);

    HAL_NVIC_SetPriority(DMA1_Stream0_IRQn, 5, 5);
    HAL_NVIC_EnableIRQ(DMA1_Stream0_IRQn);

    uint32_t start_time, dma_time, uart_time;

    // Initialize data buffer
    for (int i = 0; i < SIZE; i++) {
        data[i] = i % 255;
    }

    // DMA Transfer
    start_timer();
    start_time = get_time();
    HAL_UART_Transmit_DMA(&uart, data, SIZE);

    // Wait for DMA completion

    dma_time = get_time() - start_time;
    stop_timer();

    // Set LED states to indicate completion
    HAL_GPIO_WritePin(GPIOD, GPIO_PIN_3, GPIO_PIN_RESET);
    HAL_GPIO_WritePin(GPIOJ, GPIO_PIN_2, GPIO_PIN_SET);
    HAL_GPIO_WritePin(GPIOI, GPIO_PIN_13, GPIO_PIN_SET);

    // UART Transfer
    start_timer();
    start_time = get_time();
    HAL_UART_Transmit(&uart, data, SIZE, HAL_MAX_DELAY);
    uart_time = get_time() - start_time;
    stop_timer();

  while (1) {
  }
}

void DMA1_Stream0_IRQHandler(void) {
    HAL_DMA_IRQHandler(&dma1_struct);

    if (dma1_struct.State == HAL_DMA_STATE_READY) {
        uart.gState = HAL_UART_STATE_READY;
    }
}

/**
  * @brief System Clock Configuration
  * @retval None
  */
void SystemClock_Config(void)
{
  RCC_OscInitTypeDef RCC_OscInitStruct = {0};
  RCC_ClkInitTypeDef RCC_ClkInitStruct = {0};

  /** Supply configuration update enable
  */
  HAL_PWREx_ConfigSupply(PWR_LDO_SUPPLY);

  /** Configure the main internal regulator output voltage
  */
  __HAL_PWR_VOLTAGESCALING_CONFIG(PWR_REGULATOR_VOLTAGE_SCALE3);

  while(!__HAL_PWR_GET_FLAG(PWR_FLAG_VOSRDY)) {}

  /** Initializes the RCC Oscillators according to the specified parameters
  * in the RCC_OscInitTypeDef structure.
  */
  RCC_OscInitStruct.OscillatorType = RCC_OSCILLATORTYPE_HSI;
  RCC_OscInitStruct.HSIState = RCC_HSI_DIV1;
  RCC_OscInitStruct.HSICalibrationValue = RCC_HSICALIBRATION_DEFAULT;
  RCC_OscInitStruct.PLL.PLLState = RCC_PLL_NONE;
  if (HAL_RCC_OscConfig(&RCC_OscInitStruct) != HAL_OK)
  {
    Error_Handler();
  }

  /** Initializes the CPU, AHB and APB buses clocks
  */
  RCC_ClkInitStruct.ClockType = RCC_CLOCKTYPE_HCLK|RCC_CLOCKTYPE_SYSCLK
                              |RCC_CLOCKTYPE_PCLK1|RCC_CLOCKTYPE_PCLK2
                              |RCC_CLOCKTYPE_D3PCLK1|RCC_CLOCKTYPE_D1PCLK1;
  RCC_ClkInitStruct.SYSCLKSource = RCC_SYSCLKSOURCE_HSI;
  RCC_ClkInitStruct.SYSCLKDivider = RCC_SYSCLK_DIV1;
  RCC_ClkInitStruct.AHBCLKDivider = RCC_HCLK_DIV1;
  RCC_ClkInitStruct.APB3CLKDivider = RCC_APB3_DIV1;
  RCC_ClkInitStruct.APB1CLKDivider = RCC_APB1_DIV1;
  RCC_ClkInitStruct.APB2CLKDivider = RCC_APB2_DIV1;
  RCC_ClkInitStruct.APB4CLKDivider = RCC_APB4_DIV1;

  if (HAL_RCC_ClockConfig(&RCC_ClkInitStruct, FLASH_LATENCY_1) != HAL_OK)
  {
    Error_Handler();
  }
}

/* USER CODE BEGIN 4 */

/* USER CODE END 4 */

 /* MPU Configuration */

void MPU_Config(void)
{
  MPU_Region_InitTypeDef MPU_InitStruct = {0};

  /* Disables the MPU */
  HAL_MPU_Disable();

  /** Initializes and configures the Region and the memory to be protected
  */
  MPU_InitStruct.Enable = MPU_REGION_ENABLE;
  MPU_InitStruct.Number = MPU_REGION_NUMBER0;
  MPU_InitStruct.BaseAddress = 0x0;
  MPU_InitStruct.Size = MPU_REGION_SIZE_4GB;
  MPU_InitStruct.SubRegionDisable = 0x87;
  MPU_InitStruct.TypeExtField = MPU_TEX_LEVEL0;
  MPU_InitStruct.AccessPermission = MPU_REGION_NO_ACCESS;
  MPU_InitStruct.DisableExec = MPU_INSTRUCTION_ACCESS_DISABLE;
  MPU_InitStruct.IsShareable = MPU_ACCESS_SHAREABLE;
  MPU_InitStruct.IsCacheable = MPU_ACCESS_NOT_CACHEABLE;
  MPU_InitStruct.IsBufferable = MPU_ACCESS_NOT_BUFFERABLE;

  HAL_MPU_ConfigRegion(&MPU_InitStruct);
  /* Enables the MPU */
  HAL_MPU_Enable(MPU_PRIVILEGED_DEFAULT);

}

/**
  * @brief  This function is executed in case of error occurrence.
  * @retval None
  */
void Error_Handler(void)
{
  /* USER CODE BEGIN Error_Handler_Debug */
  /* User can add his own implementation to report the HAL error return state */
  __disable_irq();
  while (1)
  {
  }
  /* USER CODE END Error_Handler_Debug */
}

#ifdef  USE_FULL_ASSERT
/**
  * @brief  Reports the name of the source file and the source line number
  *         where the assert_param error has occurred.
  * @param  file: pointer to the source file name
  * @param  line: assert_param error line source number
  * @retval None
  */
void assert_failed(uint8_t *file, uint32_t line)
{
  /* USER CODE BEGIN 6 */
  /* User can add his own implementation to report the file name and line number,
     ex: printf("Wrong parameters value: file %s on line %d\r\n", file, line) */
  /* USER CODE END 6 */
}
#endif /* USE_FULL_ASSERT */
