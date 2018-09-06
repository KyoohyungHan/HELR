################################################################################
# Automatically-generated file. Do not edit!
################################################################################

# Add inputs and outputs from these tool invocations to the build variables 
CPP_SRCS += \
../src/ML.cpp \
../src/functions.cpp \
../src/main.cpp 

OBJS += \
./src/ML.o \
./src/functions.o \
./src/main.o 

CPP_DEPS += \
./src/ML.d \
./src/functions.d \
./src/main.d 


# Each subdirectory must supply rules for building sources it contributes
src/%.o: ../src/%.cpp
	@echo 'Building file: $<'
	@echo 'Invoking: Cross G++ Compiler'
	g++ -I/Users/kay/git/HEAAN/HEAAN/src -O2 -g3 -Wall -c -fmessage-length=0 -MMD -MP -MF"$(@:%.o=%.d)" -MT"$(@)" -o "$@" "$<" -pthread -std=c++11
	@echo 'Finished building: $<'
	@echo ' '


