//
//  DataHandler.hpp
//  ML_Library
//
//  Created by 9knifemi on 2024-02-29.
//
#ifndef DATAHANDLER_HPP
#define DATAHANDLER_HPP

#include <vector>
#include <string>

// Assuming a simplistic data structure for demonstration purposes.
// In a real scenario, you might use a more complex structure or third-party data structures.
struct DataPoint {
    std::vector<double> features;
    double label; // For simplicity, assuming a single label. Adjust as needed for your application.
};

class DataHandler {
public:
    DataHandler();

    // Load data from a CSV file or similar.
    bool loadData(const std::string& filePath, bool hasHeader = true);

    // Normalize the feature vectors in the dataset.
    void normalizeFeatures();

    // Split the loaded data into training and test sets.
    void splitData(float trainingRatio = 0.8);

    // Accessors to retrieve the training and test data.
    const std::vector<DataPoint>& getTrainingData() const;
    const std::vector<DataPoint>& getTestData() const;

private:
    std::vector<DataPoint> dataset;
    std::vector<DataPoint> trainingData;
    std::vector<DataPoint> testData;

    // Utility functions that might be useful.
    void shuffleData();
    double computeMean(const std::vector<double>& values);
    double computeStdDev(const std::vector<double>& values, double mean);
};

#endif // DATAHANDLER_HPP
