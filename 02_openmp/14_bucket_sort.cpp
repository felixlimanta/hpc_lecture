#include <cstdio>
#include <cstdlib>
#include <vector>

int main() {
  int n = 50;
  int range = 5;

  std::vector<int> key(n);  
  for (int i = 0; i < n; i++) {
    key[i] = rand() % range;
    printf("%d ",key[i]);
  }
  printf("\n");

  std::vector<int> bucket(range); 
  
  #pragma omp parallel for
  for (int i = 0; i < range; i++) {
    bucket[i] = 0;
  }
  
  #pragma omp parallel for shared(bucket)
  for (int i = 0; i < n; i++) {
    #pragma omp atomic update
    bucket[key[i]]++;
  }

  // Prefix sum to get boundaries
  std::vector<int> cml_bucket(range);
  for (int i = 0; i < range; i++) {
    cml_bucket[i] = (i > 0 ? cml_bucket[i - 1] : 0) + bucket[i];
  }

  // Fill keys
  #pragma omp parallel for shared(key)
  for (int i = 0; i < range; i++) {
    int begin = i > 0 ? cml_bucket[i - 1] : 0;
    int end = cml_bucket[i];

    std::fill(key.begin() + begin, key.begin() + end, i);
  }  

  for (int i = 0; i < n; i++) {
    printf("%d ", key[i]);
  }
  printf("\n");
}
