**train.csv**

Contains 163755 lines of battery tester data from 0 celsius and -10
celsius. Contains Voltage,Current,Ah,Wh,Power,Battery_Temp_degC,Time and SOC values
This is file to train neural network to learn SOC values.

**test.csv**

Contains 49656 lines of battery tester data from -20 celsius.
Does not include SOC values.
This will be used as input to neural network after training.
Neural network should give SOC over time curve as a result for this unseen dataset.

**test-compare-trueSOC.csv**

This is same than test.csv but inclues TRUE SOC values.
Neural network result will be compared to this to evaluate its performance and error.

**RESULT:**

Python code will train and test and plot predicted SOC vs true SOC over time.
