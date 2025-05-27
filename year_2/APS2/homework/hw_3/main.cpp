#include <algorithm>
#include <iostream>
#include <vector>

using namespace std;

int comparison_count = 0;
void insertionSort(vector<int> &arr, int left, int right);
void merge(vector<int> &arr, int l, int m, int r);
void timSort(vector<int> &arr, int n);

int main() {
    for (int n = 1; n <= 19; n++) {
        for (int k = 0; k < n; k++) {
            int chunk_size = 1 << (n - k);
            vector<int> test_data;

            for (int i = 0; i < (1 << k); i++) {
                for (int j = 1; j <= chunk_size; j++) {
                    test_data.push_back(j);
                }
            }

            comparison_count = 0;
            timSort(test_data, test_data.size());
            cout << comparison_count << " ";
        }
        cout << endl;
    }
    return 0;
}
void timSort(vector<int> &arr, int n) {
    const int RUN = 32;

    for (int i = 0; i < n; i += RUN)
        insertionSort(arr, i, min((i + RUN - 1), (n - 1)));

    for (int size = RUN; size < n; size = 2 * size) {
        for (int left = 0; left < n; left += 2 * size) {
            int mid = left + size - 1;
            int right = min((left + 2 * size - 1), (n - 1));

            if (mid < right)
                merge(arr, left, mid, right);
        }
    }
}

void insertionSort(vector<int> &arr, int left, int right) {
    for (int i = left + 1; i <= right; i++) {
        int key = arr[i];
        int j = i - 1;

        while (j >= left && arr[j] > key) {
            comparison_count++;
            arr[j + 1] = arr[j];
            j--;
        }
        if (j >= left)
            comparison_count++;
        arr[j + 1] = key;
    }
}

void merge(vector<int> &arr, int l, int m, int r) {
    int len1 = m - l + 1, len2 = r - m;
    vector<int> left(arr.begin() + l, arr.begin() + m + 1);
    vector<int> right(arr.begin() + m + 1, arr.begin() + r + 1);

    int i = 0, j = 0, k = l;
    while (i < len1 && j < len2) {
        comparison_count++;
        if (left[i] <= right[j]) {
            arr[k] = left[i];
            i++;
        } else {
            arr[k] = right[j];
            j++;
        }
        k++;
    }

    while (i < len1) {
        arr[k] = left[i];
        i++;
        k++;
    }
    while (j < len2) {
        arr[k] = right[j];
        j++;
        k++;
    }
}
