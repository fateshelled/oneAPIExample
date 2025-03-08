#include <Eigen/Core>
#include <algorithm>
#include <chrono>
#include <iostream>
#include <limits>
#include <memory>
#include <numeric>
#include <thread>
#include <vector>

#include "nanoflann.hpp"

#if __has_include(<sycl/sycl.hpp>)
#include <sycl/sycl.hpp>
#else
#include <CL/sycl.hpp>
#endif

// Structure to store K nearest neighbors and their distances
struct KNNResult {
  std::vector<std::vector<int>> indices;      // Indices of K nearest points for each query point
  std::vector<std::vector<float>> distances;  // Squared distances to K nearest points for each query point
};

// Node structure for KD-Tree (ignoring w component)
struct FlatKDNode4f {
  Eigen::Vector4f point;  // Point coordinates (w is assumed to be 1.0)
  int idx;           // Index of the point in the original dataset
  int axis;          // Split axis (0=x, 1=y, 2=z)
  int left;          // Index of left child node (-1 if none)
  int right;         // Index of right child node (-1 if none)
};

using PointCloud = std::vector<Eigen::Vector4f, Eigen::aligned_allocator<Eigen::Vector4f>>;
using KDTree = std::vector<FlatKDNode4f>;

// CPU brute force search for K nearest neighbors
KNNResult findKNearestNeighbors_bruteforce_cpu(
  const PointCloud& queries,  // Query points
  const PointCloud& points,   // Dataset points
  const int k) {              // Number of neighbors to find

  const size_t n = points.size();   // Number of dataset points
  const size_t q = queries.size();  // Number of query points

  // Initialize result structure
  KNNResult result;
  result.indices.resize(q);
  result.distances.resize(q);

  for (size_t i = 0; i < q; ++i) {
    result.indices[i].resize(k, -1);
    result.distances[i].resize(k, std::numeric_limits<float>::max());
  }

// For each query point, find K nearest neighbors
#pragma omp parallel for
  for (size_t i = 0; i < q; ++i) {
    const auto& query = queries[i];

    // Vector to store distances and indices of all points
    std::vector<std::pair<float, int>> distances(n);

    // Calculate distances to all dataset points
    for (size_t j = 0; j < n; ++j) {
      const auto& point = points[j];
      const float dx = query.x() - point.x();
      const float dy = query.y() - point.y();
      const float dz = query.z() - point.z();
      const float dist = dx * dx + dy * dy + dz * dz;
      distances[j] = {dist, j};
    }

    // Sort to find K smallest distances
    std::partial_sort(distances.begin(), distances.begin() + k, distances.end());

    // Store the results
    for (int j = 0; j < k; ++j) {
      result.indices[i][j] = distances[j].second;
      result.distances[i][j] = distances[j].first;
    }
  }

  return result;
}

KDTree createFlatKDTree4f(const PointCloud& points) {
  const size_t n = points.size();

  // Estimate tree size with some margin
  const int estimatedSize = n * 2;
  KDTree flatTree(estimatedSize);

  // Main index array
  std::vector<int> indices(n);
  std::iota(indices.begin(), indices.end(), 0);

  // Reusable temporary array for sorting
  std::vector<std::pair<float, int>> sortedValues(n);

  // Data structure for non-recursive KD-tree construction
  struct BuildTask {
    std::vector<int> indices;  // Indices corresponding to this node
    int nodeIdx;               // Node index in the tree
    int depth;                 // Depth in the tree
  };

  std::vector<BuildTask> taskStack;
  int nextNodeIdx = 1;  // Node 0 is root, subsequent nodes start from 1

  // Add the first task to the stack
  taskStack.push_back({indices, 0, 0});

  // Process until task stack is empty
  while (!taskStack.empty()) {
    // Pop a task from the stack
    BuildTask task = std::move(taskStack.back());
    taskStack.pop_back();

    std::vector<int>& subIndices = task.indices;
    const int nodeIdx = task.nodeIdx;
    const int depth = task.depth;

    if (subIndices.empty()) continue;

    // Split axis based on current depth (x->y->z->x->...)
    const int axis = depth % 3;

    // Create pairs of values and indices for sorting along the axis
    sortedValues.resize(subIndices.size());
    for (size_t i = 0; i < subIndices.size(); ++i) {
      const int idx = subIndices[i];
      sortedValues[i] = {points[idx](axis), idx};
    }

    // Partial sort to find median
    const size_t medianPos = subIndices.size() / 2;
    std::nth_element(sortedValues.begin(), sortedValues.begin() + medianPos, sortedValues.end());

    // Get the median point
    const int pointIdx = sortedValues[medianPos].second;

    // Initialize flat node
    FlatKDNode4f& node = flatTree[nodeIdx];
    node.point = points[pointIdx];
    node.idx = pointIdx;
    node.axis = axis;
    node.left = -1;
    node.right = -1;

    // Extract indices for left subtree
    std::vector<int> leftIndices;
    leftIndices.reserve(medianPos);
    for (size_t i = 0; i < medianPos; ++i) {
      leftIndices.push_back(sortedValues[i].second);
    }

    // Extract indices for right subtree
    std::vector<int> rightIndices;
    rightIndices.reserve(subIndices.size() - medianPos - 1);
    for (size_t i = medianPos + 1; i < sortedValues.size(); ++i) {
      rightIndices.push_back(sortedValues[i].second);
    }

    // Add left subtree to processing queue if not empty
    if (!leftIndices.empty()) {
      const int leftNodeIdx = nextNodeIdx++;
      node.left = leftNodeIdx;
      taskStack.push_back({std::move(leftIndices), leftNodeIdx, depth + 1});
    }

    // Add right subtree to processing queue if not empty
    if (!rightIndices.empty()) {
      const int rightNodeIdx = nextNodeIdx++;
      node.right = rightNodeIdx;
      taskStack.push_back({std::move(rightIndices), rightNodeIdx, depth + 1});
    }
  }

  // Trim the tree to actual used size
  flatTree.resize(nextNodeIdx);

  return flatTree;
}

// K nearest neighbor search using FlatKDTree on CPU
KNNResult findKNearestNeighbors_kdtree_cpu(
  const KDTree& flatTree,     // Pre-built KD-Tree
  const PointCloud& points,   // Original dataset points
  const PointCloud& queries,  // Query points
  const int k) {              // Number of neighbors to find

  const size_t n = points.size();   // Number of dataset points
  const size_t q = queries.size();  // Number of query points

  // Initialize result structure
  KNNResult result;
  result.indices.resize(q);
  result.distances.resize(q);

  for (size_t i = 0; i < q; ++i) {
    result.indices[i].resize(k, -1);
    result.distances[i].resize(k, std::numeric_limits<float>::max());
  }

// Process each query point in parallel
#pragma omp parallel for
  for (size_t queryIdx = 0; queryIdx < q; ++queryIdx) {
    const auto& query = queries[queryIdx];

    // Arrays to store distances and indices of K nearest points
    std::vector<float> bestDists(k, std::numeric_limits<float>::max());
    std::vector<int> bestIdxs(k, -1);

    // Non-recursive KD-Tree traversal using a stack
    std::vector<int> nodeStack;
    nodeStack.reserve(32);  // Reserve space but don't overallocate

    // Start from root node
    nodeStack.push_back(0);

    while (!nodeStack.empty()) {
      const int nodeIdx = nodeStack.back();
      nodeStack.pop_back();

      // Skip invalid nodes
      if (nodeIdx == -1) continue;

      const FlatKDNode4f& node = flatTree[nodeIdx];

      // Calculate distance to current node (3D space)
      const float dx = query.x() - node.point.x();
      const float dy = query.y() - node.point.y();
      const float dz = query.z() - node.point.z();
      const float dist = dx * dx + dy * dy + dz * dz;

      // Check if this node should be included in K nearest
      if (dist < bestDists[k - 1]) {
        // Simple linear insertion - works well for small k
        int insertPos = k - 1;
        while (insertPos > 0 && dist < bestDists[insertPos - 1]) {
          bestDists[insertPos] = bestDists[insertPos - 1];
          bestIdxs[insertPos] = bestIdxs[insertPos - 1];
          insertPos--;
        }

        // Insert new point
        bestDists[insertPos] = dist;
        bestIdxs[insertPos] = node.idx;
      }

      // Distance along split axis
      float axisDistance;
      if (node.axis == 0)
        axisDistance = dx;
      else if (node.axis == 1)
        axisDistance = dy;
      else
        axisDistance = dz;

      // Determine nearer and further subtrees
      const int nearerNode = (axisDistance <= 0) ? node.left : node.right;
      const int furtherNode = (axisDistance <= 0) ? node.right : node.left;

      // Add nodes to stack in reverse order of traversal (nearer one on top)
      if (axisDistance * axisDistance <= bestDists[k - 1]) {
        if (furtherNode != -1) {
          nodeStack.push_back(furtherNode);
        }
      }

      if (nearerNode != -1) {
        nodeStack.push_back(nearerNode);
      }
    }

    // Store results
    for (int i = 0; i < k; i++) {
      result.indices[queryIdx][i] = bestIdxs[i];
      result.distances[queryIdx][i] = bestDists[i];
    }
  }

  return result;
}

template <size_t MAX_K = 50>
KNNResult findKNearestNeighbors_bruteforce_sycl(sycl::queue& queue, const PointCloud& queries, const PointCloud& points, const int k) {
  const size_t n = points.size();   // Number of dataset points
  const size_t q = queries.size();  // Number of query points

  // Initialize result structure
  KNNResult result;
  result.indices.resize(q);
  result.distances.resize(q);

  for (size_t i = 0; i < q; ++i) {
    result.indices[i].resize(k, -1);
    result.distances[i].resize(k, std::numeric_limits<float>::max());
  }

  try {
    // Allocate device memory using USM
    Eigen::Vector4f* d_points = sycl::malloc_device<Eigen::Vector4f>(n, queue);
    Eigen::Vector4f* d_queries = sycl::malloc_device<Eigen::Vector4f>(q, queue);
    float* d_distances = sycl::malloc_device<float>(q * k, queue);
    int* d_neighbors = sycl::malloc_device<int>(q * k, queue);

    // Copy to device memory
    queue.memcpy(d_points, points[0].data(), n * sizeof(Eigen::Vector4f));
    queue.memcpy(d_queries, queries[0].data(), q * sizeof(Eigen::Vector4f));

    // Initialize distances and neighbor lists
    queue.fill(d_distances, std::numeric_limits<float>::max(), q * k);
    queue.fill(d_neighbors, -1, q * k);
    queue.wait();  // Wait for copy and initialization to complete

    // KNN search kernel
    queue
      .submit([&](sycl::handler& h) {
        h.parallel_for(sycl::range<1>(q), [=](sycl::id<1> idx) {
          const size_t queryIdx = idx[0];
          const auto query = d_queries[queryIdx];

          // Arrays to store K nearest points
          float kDistances[MAX_K];
          int kIndices[MAX_K];

          // Initialize
          for (int i = 0; i < k; i++) {
            kDistances[i] = std::numeric_limits<float>::max();
            kIndices[i] = -1;
          }

          // Calculate distances to all dataset points
          for (size_t j = 0; j < n; j++) {
            // Calculate 3D distance
            const float dx = query.x() - d_points[j].x();
            const float dy = query.y() - d_points[j].y();
            const float dz = query.z() - d_points[j].z();
            const float dist = dx * dx + dy * dy + dz * dz;

            // Check if this point should be included in K nearest
            if (dist < kDistances[k - 1]) {
              // Find insertion position
              int insertPos = k - 1;
              while (insertPos > 0 && dist < kDistances[insertPos - 1]) {
                kDistances[insertPos] = kDistances[insertPos - 1];
                kIndices[insertPos] = kIndices[insertPos - 1];
                insertPos--;
              }

              // Insert new point
              kDistances[insertPos] = dist;
              kIndices[insertPos] = j;
            }
          }

          // Write results to global memory
          for (int i = 0; i < k; i++) {
            d_distances[queryIdx * k + i] = kDistances[i];
            d_neighbors[queryIdx * k + i] = kIndices[i];
          }
        });
      })
      .wait();

    // Copy results back to host memory
    float* h_distances = sycl::malloc_host<float>(q * k, queue);
    int* h_neighbors = sycl::malloc_host<int>(q * k, queue);

    queue.memcpy(h_distances, d_distances, q * k * sizeof(float));
    queue.memcpy(h_neighbors, d_neighbors, q * k * sizeof(int));
    queue.wait();

    // Copy to output structure
    for (size_t i = 0; i < q; i++) {
      for (int j = 0; j < k; j++) {
        result.indices[i][j] = h_neighbors[i * k + j];
        result.distances[i][j] = h_distances[i * k + j];
      }
    }

    // Free USM memory
    sycl::free(d_points, queue);
    sycl::free(d_queries, queue);
    sycl::free(d_distances, queue);
    sycl::free(d_neighbors, queue);
    sycl::free(h_distances, queue);
    sycl::free(h_neighbors, queue);
  } catch (const sycl::exception& e) {
    std::cerr << "SYCL exception caught: " << e.what() << std::endl;
  }

  return result;
}

template <size_t MAX_K = 50>
KNNResult findKNearestNeighbors_kdtree_sycl(
  sycl::queue& queue,         // SYCL execution queue
  const KDTree& flatTree,     // Pre-built KD-Tree
  const PointCloud& points,   // Original dataset points
  const PointCloud& queries,  // Query points
  const int k) {              // Number of neighbors to find

  const size_t n = points.size();   // Number of dataset points
  const size_t q = queries.size();  // Number of query points
  const size_t treeSize = flatTree.size();

  // Initialize result structure
  KNNResult result;
  result.indices.resize(q);
  result.distances.resize(q);

  for (size_t i = 0; i < q; ++i) {
    result.indices[i].resize(k, -1);
    result.distances[i].resize(k, std::numeric_limits<float>::max());
  }

  try {
    // Allocate device memory using USM
    FlatKDNode4f* d_tree = sycl::malloc_device<FlatKDNode4f>(treeSize, queue);
    Eigen::Vector4f* d_queries = sycl::malloc_device<Eigen::Vector4f>(q, queue);
    float* s_distances = sycl::malloc_shared<float>(q * k, queue);
    int* s_neighbors = sycl::malloc_shared<int>(q * k, queue);

    // Copy to device memory
    queue.memcpy(d_tree, flatTree.data(), treeSize * sizeof(FlatKDNode4f));
    queue.memcpy(d_queries, queries[0].data(), q * sizeof(Eigen::Vector4f));

    // Initialize distances and neighbor lists
    queue.fill(s_distances, std::numeric_limits<float>::max(), q * k);
    queue.fill(s_neighbors, -1, q * k);
    queue.wait();  // Wait for copy and initialization to complete

    // SYCL KD-Tree KNN search kernel
    queue
      .submit([&](sycl::handler& h) {
        h.parallel_for(sycl::range<1>(q), [=](sycl::id<1> idx) {
          const size_t queryIdx = idx[0];
          // Query point
          const auto& query = d_queries[queryIdx];

          // Arrays to store K nearest points
          float bestDists[MAX_K];
          int bestIdxs[MAX_K];

          // Initialize
          for (int i = 0; i < k; i++) {
            bestDists[i] = std::numeric_limits<float>::max();
            bestIdxs[i] = -1;
          }

          // Non-recursive KD-Tree traversal using a stack
          const int MAX_DEPTH = 32;  // Maximum stack depth
          int nodeStack[MAX_DEPTH];
          int stackPtr = 0;

          // Start from root node
          nodeStack[stackPtr++] = 0;

          while (stackPtr > 0) {
            const int nodeIdx = nodeStack[--stackPtr];

            // Skip invalid nodes
            if (nodeIdx == -1) continue;

            const FlatKDNode4f& node = d_tree[nodeIdx];

            // Calculate distance to current node (3D space)
            const float dx = query.x() - node.point.x();
            const float dy = query.y() - node.point.y();
            const float dz = query.z() - node.point.z();
            const float dist = dx * dx + dy * dy + dz * dz;

            // Check if this node should be included in K nearest
            if (dist < bestDists[k - 1]) {
              // Find insertion position
              int insertPos = k - 1;
              while (insertPos > 0 && dist < bestDists[insertPos - 1]) {
                bestDists[insertPos] = bestDists[insertPos - 1];
                bestIdxs[insertPos] = bestIdxs[insertPos - 1];
                insertPos--;
              }

              // Insert new point
              bestDists[insertPos] = dist;
              bestIdxs[insertPos] = node.idx;
            }

            // Distance along split axis
            const float axisDistance (node.axis == 0 ? dx : (node.axis == 1 ? dy : dz));

            // Determine nearer and further subtrees
            const int nearerNode = (axisDistance <= 0) ? node.left : node.right;
            const int furtherNode = (axisDistance <= 0) ? node.right : node.left;

            // If distance to splitting plane is less than current kth distance,
            // both subtrees must be searched
            if (axisDistance * axisDistance <= bestDists[k - 1]) {
              // Add further subtree to stack
              if (furtherNode != -1 && stackPtr < MAX_DEPTH) {
                nodeStack[stackPtr++] = furtherNode;
              }
            }

            // Add nearer subtree to stack
            if (nearerNode != -1 && stackPtr < MAX_DEPTH) {
              nodeStack[stackPtr++] = nearerNode;
            }
          }

          // Write results to global memory
          for (int i = 0; i < k; i++) {
            s_distances[queryIdx * k + i] = bestDists[i];
            s_neighbors[queryIdx * k + i] = bestIdxs[i];
          }
        });
      })
      .wait();

    // Copy to output structure
    for (size_t i = 0; i < q; i++) {
      for (int j = 0; j < k; j++) {
        result.indices[i][j] = s_neighbors[i * k + j];
        result.distances[i][j] = s_distances[i * k + j];
      }
    }

    // Free USM memory
    sycl::free(d_tree, queue);
    sycl::free(d_queries, queue);
    sycl::free(s_distances, queue);
    sycl::free(s_neighbors, queue);
  } catch (const sycl::exception& e) {
    std::cerr << "SYCL exception caught: " << e.what() << std::endl;
  }

  return result;
}

// Adapter class for nanoflann
template <typename T>
struct PointCloudAdapter {
  const std::vector<Eigen::Vector4<T>, Eigen::aligned_allocator<Eigen::Vector4<T>>>& points;

  PointCloudAdapter(const std::vector<Eigen::Vector4<T>, Eigen::aligned_allocator<Eigen::Vector4<T>>>& pts) : points(pts) {}

  inline size_t kdtree_get_point_count() const { return points.size(); }

  inline T kdtree_get_pt(const size_t idx, const size_t dim) const { return points[idx](dim); }

  template <class BBOX>
  bool kdtree_get_bbox(BBOX&) const {
    return false;
  }
};

KNNResult findKNearestNeighbors_nanoflann(
  const PointCloud& queries,  // Query points
  const PointCloud& points,   // Original dataset points
  const int k) {              // Number of neighbors to find

  const size_t n = points.size();   // Number of dataset points
  const size_t q = queries.size();  // Number of query points

  // Initialize result structure
  KNNResult result;
  result.indices.resize(q);
  result.distances.resize(q);

  for (size_t i = 0; i < q; ++i) {
    result.indices[i].resize(k, -1);
    result.distances[i].resize(k, std::numeric_limits<float>::max());
  }

  try {
    // define Adapter and KDTree
    typedef PointCloudAdapter<float> PCAdapt;
    PCAdapt pcadapt(points);
    typedef nanoflann::KDTreeSingleIndexAdaptor<nanoflann::L2_Simple_Adaptor<float, PCAdapt>, PCAdapt, 3> KDTreeType;

    // build KDTree
    KDTreeType index(3, pcadapt, nanoflann::KDTreeSingleIndexAdaptorParams(10 /* max leaf */));
    index.buildIndex();

    std::vector<size_t> ret_indices(k);

    // KNN search
#pragma omp parallel for
    for (size_t i = 0; i < q; ++i) {
      nanoflann::KNNResultSet<float> resultSet(k);
      resultSet.init(&ret_indices[0], &result.distances[i][0]);
      index.findNeighbors(resultSet, queries[i].data(), nanoflann::SearchParameters());

      // Copy to output structure
      for (size_t j = 0; j < k; ++j) {
        result.indices[i][j] = ret_indices[j];
      }
    }

  } catch (const std::exception& e) {
    std::cerr << "nanoflann exception caught: " << e.what() << std::endl;
  }

  return result;
}

// Main function
int main() {
  const size_t LOOP = 10;
  const size_t knn = 10;

  // Generate random 3D points
  const size_t numTargetPoints = 128 * 256;
  PointCloud target_points;
  target_points.reserve(numTargetPoints);

  const size_t numQueryPoints = 128 * 256;
  PointCloud query_points;
  query_points.reserve(numQueryPoints);

  srand(42);  // Use reproducible random numbers
  for (size_t i = 0; i < numTargetPoints; ++i) {
    const float x = static_cast<float>(rand()) / static_cast<float>(RAND_MAX) * 100.0f;
    const float y = static_cast<float>(rand()) / static_cast<float>(RAND_MAX) * 100.0f;
    const float z = static_cast<float>(rand()) / static_cast<float>(RAND_MAX) * 100.0f;
    target_points.push_back(Eigen::Vector4f(x, y, z, 1.0));
  }

  for (size_t i = 0; i < numQueryPoints; ++i) {
    const float x = static_cast<float>(rand()) / static_cast<float>(RAND_MAX) * 100.0f;
    const float y = static_cast<float>(rand()) / static_cast<float>(RAND_MAX) * 100.0f;
    const float z = static_cast<float>(rand()) / static_cast<float>(RAND_MAX) * 100.0f;
    query_points.push_back(Eigen::Vector4f(x, y, z, 1.0));
  }

  std::cout << "Target points: " << target_points.size() << std::endl;
  std::cout << "Query points: " << query_points.size() << std::endl;

  // Create SYCL queue
  sycl::queue queue(sycl::default_selector_v);
  std::cout << "Device: " << queue.get_device().get_info<sycl::info::device::name>() << std::endl;

  // CPU brute force search (reference implementation)
  std::cout << "Running CPU brute force search (reference)..." << std::endl;
  double elapsed_cpu_bruteforce = 0.0;
  KNNResult nearestCPUBruteForce;
  for (size_t i = 0; i < 1; ++i) {  // Only run once as it might be slow
    auto start_cpu_bruteforce = std::chrono::high_resolution_clock::now();
    nearestCPUBruteForce = findKNearestNeighbors_bruteforce_cpu(query_points, target_points, knn);
    auto end_cpu_bruteforce = std::chrono::high_resolution_clock::now();
    elapsed_cpu_bruteforce += std::chrono::duration_cast<std::chrono::milliseconds>(end_cpu_bruteforce - start_cpu_bruteforce).count();
  }
  std::cout << "CPU brute force search completed in " << elapsed_cpu_bruteforce << " ms." << std::endl;

  // CPU KDTree
  std::cout << "Running CPU KD-Tree search..." << std::endl;
  double elapsed_cpu_kdtree = 0.0;
  KNNResult nearestCPUKDTree;
  for (size_t i = 0; i < LOOP; ++i) {
    auto start_cpu_kdtree = std::chrono::high_resolution_clock::now();
    const auto kdtree = createFlatKDTree4f(target_points);
    nearestCPUKDTree = findKNearestNeighbors_kdtree_cpu(kdtree, target_points, query_points, knn);
    auto end_cpu_kdtree = std::chrono::high_resolution_clock::now();
    elapsed_cpu_kdtree += std::chrono::duration_cast<std::chrono::milliseconds>(end_cpu_kdtree - start_cpu_kdtree).count();
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
  }
  elapsed_cpu_kdtree /= LOOP;
  std::cout << "CPU KD-Tree search completed." << std::endl;

  // SYCL brute force search
  std::cout << "Running SYCL brute force search..." << std::endl;
  double elapsed_sycl_bruteforce = 0.0;
  KNNResult nearestSYCLBruteForce;
  for (size_t i = 0; i < LOOP; ++i) {
    auto start_sycl_bruteforce = std::chrono::high_resolution_clock::now();
    nearestSYCLBruteForce = findKNearestNeighbors_bruteforce_sycl(queue, query_points, target_points, knn);
    auto end_sycl_bruteforce = std::chrono::high_resolution_clock::now();
    elapsed_sycl_bruteforce += std::chrono::duration_cast<std::chrono::milliseconds>(end_sycl_bruteforce - start_sycl_bruteforce).count();
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
  }
  elapsed_sycl_bruteforce /= LOOP;
  std::cout << "SYCL brute force search completed." << std::endl;

  // KD-Tree implementation with SYCL
  std::cout << "Running SYCL KD-Tree search..." << std::endl;
  double elapsed_sycl_kdtree = 0.0;
  KNNResult nearestKDTree;
  for (size_t i = 0; i < LOOP; ++i) {
    auto start_kdtree = std::chrono::high_resolution_clock::now();
    const auto kdtree = createFlatKDTree4f(target_points);
    nearestKDTree = findKNearestNeighbors_kdtree_sycl(queue, kdtree, target_points, query_points, knn);
    auto end_kdtree = std::chrono::high_resolution_clock::now();
    elapsed_sycl_kdtree += std::chrono::duration_cast<std::chrono::milliseconds>(end_kdtree - start_kdtree).count();
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
  }
  elapsed_sycl_kdtree /= LOOP;
  std::cout << "SYCL KD-Tree search completed." << std::endl;

  // nanoflann KD-Tree
  std::cout << "Running nanoflann KD-Tree search..." << std::endl;
  double elapsed_nanoflann = 0.0;
  KNNResult nearestNanoflann;
  for (size_t i = 0; i < LOOP; ++i) {
    auto start_nanoflann = std::chrono::high_resolution_clock::now();
    nearestNanoflann = findKNearestNeighbors_nanoflann(query_points, target_points, knn);
    auto end_nanoflann = std::chrono::high_resolution_clock::now();
    elapsed_nanoflann += std::chrono::duration_cast<std::chrono::milliseconds>(end_nanoflann - start_nanoflann).count();
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
  }
  elapsed_nanoflann /= LOOP;
  std::cout << "nanoflann KD-Tree search completed." << std::endl;

  // Verify CPU KDTree against CPU brute force
  bool cpuKdtreeCorrect = true;
  int mismatchCount = 0;
  for (size_t i = 0; i < query_points.size(); ++i) {
    for (size_t j = 0; j < knn; ++j) {
      if (nearestCPUBruteForce.indices[i][j] != nearestCPUKDTree.indices[i][j]) {
        // Check if distance is the same (may have different indices with same distance)
        float distBrute = nearestCPUBruteForce.distances[i][j];
        float distKDTree = nearestCPUKDTree.distances[i][j];
        if (std::abs(distBrute - distKDTree) > 1e-5f) {
          ++mismatchCount;
          if (mismatchCount < 10) {
            std::cout << "CPU KD-Tree mismatch at query " << i << ", neighbor " << j << ": CPU Brute idx=" << nearestCPUBruteForce.indices[i][j] << " (dist=" << distBrute
                      << "), CPU KD-Tree idx=" << nearestCPUKDTree.indices[i][j] << " (dist=" << distKDTree << ")" << std::endl;
          }
          cpuKdtreeCorrect = false;
        }
      }
    }
  }

  // Verify SYCL brute force against CPU brute force
  std::cout << "\nVerifying SYCL brute force against CPU reference..." << std::endl;
  bool syclBruteForceCorrect = true;
  mismatchCount = 0;
  for (size_t i = 0; i < query_points.size(); ++i) {
    for (size_t j = 0; j < knn; ++j) {
      if (nearestCPUBruteForce.indices[i][j] != nearestSYCLBruteForce.indices[i][j]) {
        // Check if distance is the same (may have different indices with same distance)
        float distCPU = nearestCPUBruteForce.distances[i][j];
        float distSYCL = nearestSYCLBruteForce.distances[i][j];
        if (std::abs(distCPU - distSYCL) > 1e-5f) {
          ++mismatchCount;
          if (mismatchCount < 10) {
            std::cout << "SYCL Brute Force mismatch at query " << i << ", neighbor " << j << ": CPU idx=" << nearestCPUBruteForce.indices[i][j] << " (dist=" << distCPU
                      << "), SYCL idx=" << nearestSYCLBruteForce.indices[i][j] << " (dist=" << distSYCL << ")" << std::endl;
          }
          syclBruteForceCorrect = false;
        }
      }
    }
  }

  // Verify KD-Tree against CPU brute force
  std::cout << "\nVerifying SYCL KD-Tree against CPU reference..." << std::endl;
  bool kdtreeCorrect = true;
  mismatchCount = 0;
  for (size_t i = 0; i < query_points.size(); ++i) {
    for (size_t j = 0; j < knn; ++j) {
      if (nearestCPUBruteForce.indices[i][j] != nearestKDTree.indices[i][j]) {
        // Check if distance is the same (may have different indices with same distance)
        float distCPU = nearestCPUBruteForce.distances[i][j];
        float distKDTree = nearestKDTree.distances[i][j];
        if (std::abs(distCPU - distKDTree) > 1e-5f) {
          ++mismatchCount;
          if (mismatchCount < 10) {
            std::cout << "KD-Tree mismatch at query " << i << ", neighbor " << j << ": CPU idx=" << nearestCPUBruteForce.indices[i][j] << " (dist=" << distCPU
                      << "), KD-Tree idx=" << nearestKDTree.indices[i][j] << " (dist=" << distKDTree << ")" << std::endl;
          }
          kdtreeCorrect = false;
        }
      }
    }
  }

  // Verify KD-Tree against nanoflann
  bool nanoflannCorrect = true;
  mismatchCount = 0;
  for (size_t i = 0; i < query_points.size(); ++i) {
    for (size_t j = 0; j < knn; ++j) {
      if (nearestCPUBruteForce.indices[i][j] != nearestNanoflann.indices[i][j]) {
        // 距離が同じかどうか確認
        float distCPU = nearestCPUBruteForce.distances[i][j];
        float distNanoflann = nearestNanoflann.distances[i][j];
        if (std::abs(distCPU - distNanoflann) > 1e-5f) {
          ++mismatchCount;
          if (mismatchCount < 10) {
            std::cout << "nanoflann mismatch at query " << i << ", neighbor " << j << ": CPU idx=" << nearestCPUBruteForce.indices[i][j] << " (dist=" << distCPU
                      << "), nanoflann idx=" << nearestNanoflann.indices[i][j] << " (dist=" << distNanoflann << ")" << std::endl;
          }
          nanoflannCorrect = false;
        }
      }
    }
  }

  // Output performance results
  std::cout << "\nPerformance Results:" << std::endl;
  std::cout << "CPU Brute Force:  " << elapsed_cpu_bruteforce << " ms" << std::endl;
  std::cout << "CPU KD-Tree:       " << elapsed_cpu_kdtree << " ms" << std::endl;
  std::cout << "SYCL Brute Force: " << elapsed_sycl_bruteforce << " ms" << std::endl;
  std::cout << "SYCL KD-Tree:     " << elapsed_sycl_kdtree << " ms" << std::endl;
  std::cout << "nanoflann KD-Tree:  " << elapsed_nanoflann << " ms" << std::endl;

  std::cout << "CPU Brute Force vs CPU KD-Tree Speedup: " << elapsed_cpu_bruteforce / elapsed_cpu_kdtree << "x" << std::endl;
  std::cout << "CPU Brute Force vs SYCL Brute Force Speedup: " << elapsed_cpu_bruteforce / elapsed_sycl_bruteforce << "x" << std::endl;
  std::cout << "CPU KD-Tree vs SYCL KD-Tree Speedup: " << elapsed_cpu_kdtree / elapsed_sycl_kdtree << "x" << std::endl;
  std::cout << "SYCL Brute Force vs SYCL KD-Tree Speedup: " << elapsed_sycl_bruteforce / elapsed_sycl_kdtree << "x" << std::endl;
  std::cout << "SYCL KD-Tree vs nanoflann KD-Tree: " << elapsed_sycl_kdtree / elapsed_nanoflann << "x" << std::endl;
  std::cout << "CPU vs nanoflann KD-Tree:          " << elapsed_cpu_bruteforce / elapsed_nanoflann << "x" << std::endl;

  // Output verification results
  std::cout << "\nAccuracy Check:" << std::endl;
  std::cout << "CPU KD-Tree correct:        " << (cpuKdtreeCorrect ? "Yes" : "No") << std::endl;
  std::cout << "SYCL Brute Force correct:  " << (syclBruteForceCorrect ? "Yes" : "No") << std::endl;
  std::cout << "SYCL KD-Tree correct:      " << (kdtreeCorrect ? "Yes" : "No") << std::endl;
  std::cout << "nanoflann KD-Tree correct:   " << (nanoflannCorrect ? "Yes" : "No") << std::endl;

  if (!syclBruteForceCorrect || !kdtreeCorrect) {
    std::cout << "\nNOTE: Some mismatches were found, but this may be acceptable if:" << std::endl;
    std::cout << "1. The distances are very close (floating point precision differences)" << std::endl;
    std::cout << "2. Different points have exactly the same distance from a query point" << std::endl;
    std::cout << "   (in this case, the order can be arbitrary)" << std::endl;
  }

  return 0;
}
