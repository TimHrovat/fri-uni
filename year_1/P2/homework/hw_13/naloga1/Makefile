
# Pred poganjanjem programa make:
#     export name=<ime_datoteke_C_brez_koncnice>
#
# Prevajanje:
#     make
#
# Testiranje:
#     make test [neobvezni parametri]
#
#     Neobvezni parametri:
#         T: "casovna omejitev v sekundah (privzeto = 1)
#         M: za vsak testni primer izpi"si "cas izvajanja (1: da; 0: ne; privzeto = 0)
#         P: obseg testnih primerov, ki naj se obravnava (npr. 3-5 ali 7 ali 2- ali -6 ali -; privzeto = -)
#
#     Primeri:
#         make test
#         make test P=4-7
#         make test T=3
#         make test T=2 M=1 P=6-

CC = gcc
CFLAGS = -std=c99 -pedantic -Wall
LIBS = -lm

T = 1
M = 0
P = '-'

DIFF_TIMEOUT = 0.5

TIMEOUT_SIGNAL = 124
SHELL = /bin/bash
OK_STRING=\033[1;32mOK\033[0;38m
FAILED_STRING=\033[1;31mfailed\033[0;38m
TIMEOUT_STRING=\033[1;35mtimeout\033[0;38m
TEMPORARY_OUTPUT=temp.out

${name} : ${name}.c
	@echo "=== COMPILING: ==="
	${CC} ${CFLAGS} -o $@ $< ${LIBS}

test : ${name}
	@echo "=== TESTING: ==="; \
	singleTestFile=$$(ls -1 test*.par | head -1); \
	\
	testNumberLength=$$(( $${#singleTestFile} - 8 )); \
	cutParameter=$$((5 + testNumberLength - 1)); \
	nTests=$$(ls -1 test*.par | cut -c5-$$cutParameter | sort | tail -1); \
	firstTest=$$(echo $$P | cut -d'-' -f1); \
	lastTest=$$(echo $$P | cut -d'-' -f2); \
	if [[ -z $$firstTest ]]; then \
		firstTest=1; \
	fi; \
	if [[ -z $$lastTest ]]; then \
		lastTest=$$nTests; \
	fi; \
	\
	nPassedTests=0; \
	nConsideredTests=0; \
	\
	trap exit SIGINT; \
	\
	timeoutValue=${T}; \
	hardTimeoutValue=$$((timeoutValue + 2)); \
	\
	for i0 in $$(seq $$firstTest $$lastTest); do \
		i=$$(printf %0$${testNumberLength}d $$i0); \
		startTime=$$(date +%s.%N); \
		timeout -k $${hardTimeoutValue} $${timeoutValue} ./${name} $$(cat test$${i}.par) 2> /dev/null; \
		result=$$?; \
		endTime=$$(date +%s.%N); \
		if [[ $$result == ${TIMEOUT_SIGNAL} ]]; then \
			if [[ ${M} = 1 ]]; then \
				echo -e "test$$i -- ${TIMEOUT_STRING}  [> $${timeoutValue} s]"; \
			else \
				echo -e "test$$i -- ${TIMEOUT_STRING}"; \
			fi; \
		else \
			timeDifference=$$(echo "scale=2; $${endTime} - $${startTime}" | bc | awk '{printf "%.2f\n", $$0}'); \
			measuredTimeString=""; \
			if [[ ${M} = 1 ]]; then \
				measuredTimeString=" [$$timeDifference s]"; \
			fi; \
			imeDatoteke=$$(cat test$${i}.par | cut -d' ' -f1); \
			kos=$$(cat test$${i}.par | cut -d' ' -f2); \
			stDatotek=$$(ls -1 $${imeDatoteke}.* | wc -l); \
			truncate -s 0 ${TEMPORARY_OUTPUT}; \
			for j in $$(seq 0 $$((stDatotek - 1))); do \
				velikost=$$(stat -f %z $${imeDatoteke}.$$j); \
				cat $${imeDatoteke}.$$j >> ${TEMPORARY_OUTPUT}; \
				if [ $$velikost -ne $$kos ]; then \
					break; \
				fi; \
			done; \
			timeout -k ${DIFF_TIMEOUT} ${DIFF_TIMEOUT} diff ${TEMPORARY_OUTPUT} $${imeDatoteke} > /dev/null; \
			result=$$?; \
			if [[ $$result == ${TIMEOUT_SIGNAL} ]]; then \
				cmp ${TEMPORARY_OUTPUT} $${imeDatoteke} > /dev/null; \
				result=$$?; \
			fi; \
			if [[ $$result == 0 ]]; then \
				echo -e "test$$i -- ${OK_STRING} $${measuredTimeString}"; \
				nPassedTests=$$((nPassedTests + 1)); \
			else \
				echo -e "test$$i -- ${FAILED_STRING} $${measuredTimeString}"; \
			fi \
		fi; \
		nConsideredTests=$$((nConsideredTests + 1)); \
	done; \
	echo "Result: $$nPassedTests / $$nConsideredTests"

clean :
	rm -fr ${name} dat*.* ${TEMPORARY_OUTPUT}
