#include <iostream>
#include <vector>
#include <chrono>
#include <iomanip>
#include <random>
#include <algorithm>

using namespace std;
using namespace std::chrono;

int numSwaps = 0;
int numComparisons = 0;

void swapWithCount(int &a, int &b);
int compareWithCount(int a, int b);
void dualPivotQuickSort(vector<int> &arr, int low, int high);
vector<int> generateRandomArray(int size);

int main() {
    vector<int> sizes;
    vector<long long> timesRandom, timesSorted;
    vector<int> swapsRandom, swapsSorted;
    vector<int> compsRandom, compsSorted;

    cout << fixed << setprecision(2);

    for (int n = 5; n <= 10; ++n) {
        int size = 1 << n;
        sizes.push_back(size);

        // Random
        vector<int> arrRand = generateRandomArray(size);
        numSwaps = numComparisons = 0;
        auto start = high_resolution_clock::now();
        dualPivotQuickSort(arrRand, 0, size - 1);
        auto end = high_resolution_clock::now();
        long long duration = duration_cast<microseconds>(end - start).count();
        timesRandom.push_back(duration);
        swapsRandom.push_back(numSwaps);
        compsRandom.push_back(numComparisons);

        // Sorted
        vector<int> arrSorted(size);
        iota(arrSorted.begin(), arrSorted.end(), 1);
        numSwaps = numComparisons = 0;
        start = high_resolution_clock::now();
        dualPivotQuickSort(arrSorted, 0, size - 1);
        end = high_resolution_clock::now();
        duration = duration_cast<microseconds>(end - start).count();
        timesSorted.push_back(duration);
        swapsSorted.push_back(numSwaps);
        compsSorted.push_back(numComparisons);
    }

    cout << "\nCas (mikrosekunde):\n";
    cout << "Velikost\tNakljucna\tUrejena\n";
    for (int i = 0; i < sizes.size(); ++i) {
        cout << sizes[i] << "\t\t" << timesRandom[i] << "\t\t" << timesSorted[i] << "\n";
    }

    cout << "\nStevilo zamenjav:\n";
    cout << "Velikost\tNakljucna\tUrejena\n";
    for (int i = 0; i < sizes.size(); ++i) {
        cout << sizes[i] << "\t\t" << swapsRandom[i] << "\t\t" << swapsSorted[i] << "\n";
    }

    cout << "\nStevilo primerjav:\n";
    cout << "Velikost\tNakljucna\tUrejena\n";
    for (int i = 0; i < sizes.size(); ++i) {
        cout << sizes[i] << "\t\t" << compsRandom[i] << "\t\t" << compsSorted[i] << "\n";
    }

    return 0;
}

void swapWithCount(int &a, int &b) {
    ++numSwaps;
    swap(a, b);
}

int compareWithCount(int a, int b) {
    ++numComparisons;
    return a - b;
}

void dualPivotQuickSort(vector<int> &arr, int low, int high) {
    if (low < high) {
        if (compareWithCount(arr[low], arr[high]) > 0)
            swapWithCount(arr[low], arr[high]);

        int lp = low + 1;
        int rp = high - 1;
        int i = lp;

        while (i <= rp) {
            if (compareWithCount(arr[i], arr[low]) < 0) {
                swapWithCount(arr[i], arr[lp]);
                ++lp;
            } else if (compareWithCount(arr[i], arr[high]) > 0) {
                while (compareWithCount(arr[rp], arr[high]) > 0 && i < rp) {
                    --rp;
                }
                swapWithCount(arr[i], arr[rp]);
                --rp;
                if (compareWithCount(arr[i], arr[low]) < 0) {
                    swapWithCount(arr[i], arr[lp]);
                    ++lp;
                }
            }
            ++i;
        }

        --lp;
        ++rp;

        swapWithCount(arr[low], arr[lp]);
        swapWithCount(arr[high], arr[rp]);

        dualPivotQuickSort(arr, low, lp - 1);
        dualPivotQuickSort(arr, lp + 1, rp - 1);
        dualPivotQuickSort(arr, rp + 1, high);
    }
}

vector<int> generateRandomArray(int size) {
    vector<int> arr(size);
    random_device rd;
    mt19937 gen(rd());
    uniform_int_distribution<> dis(1, size * 10);
    for (int &x : arr) {
        x = dis(gen);
    }
    return arr;
}

