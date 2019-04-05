# HELR: Homomorphic Logistic Regression on Encrypted Data

## How to run this program?

In 'Debug' folder, just use 'make all' command (might need to type 'make clean' first if you re-construct this program).

This will construct 'HELR' file that we can run.

The command to run HELR is follow:

  ./HELR file_name_for_train file_name_for_test isTargetFirst isEncrypted numThread

- Here files for train and test should be in data folder with csv format (distinguished by comma).
- isTargetFirst: the target value (Y_i) is at the first column or last.
- isEncrypted: usually "1", but you can use "0" if you want to check plaintext logistic regression (with approximate signmoid).
- numThread: the number of thread for multi-threading.
- If you want to change iteration number and leraning rate, see main.cpp file.

Notice that you might need to change the path to the NTL library and HEAAN library (at makefile and src/subdir.mk).

NTL lib: http://www.shoup.net/ntl/

HEAAN lib: https://github.com/kimandrik/HEAAN (commit: c2f08aa6163d7ae193f54419559d8decc6f05ef4)

## Result (with MNIST data)

We use compressed MNIST data (original data has size 28 by 28 while our data is 14 by 14). The method for compression is computing mean of 4 element in 2 by 2. The command run same experiment is "./HELR MNIST_train.txt MNIST_test.txt 1 1 8" (with 32 iteration and learning rate 1.0).

| Training Time  | AUROC | Accuracy |
| -------------- | ------------- | -----|
|    68.33 min   | 0.99 | 96.2% |

#### Testing PC information
- 32 number of Intel(R) Xeon(R) CPU E5-2620 v4 2.10 GHz (each CPU has 2 cores, we used 8 thread)
- 64GB RAM (we use about 10GB RAM in this experiment)

## Example

```c++
// Set BasicThreadPool for multi-threading
BasicThreadPool pool(numThread);

// Read Data
long factorNum, sampleNum;
double** zData = SecureML::zDataFromFile(file, factorNum, sampleNum, isFirst);

// Set parameter
long iterNum = 32;
long learningRate = 1.0;
SecureML::Params params(factorNum, sampleNum, iterNum, learningRate, pool.NumThreads());
params.path_to_file = file; ///< path for train data
params.path_to_test_file = file_test; ///< path for test result
params.isfirst = isFirst; ///< Y is at the first column of the data

// Construct and Key Gen
Ring ring(params.logN, params.logQBoot);
SecretKey sk(ring);
Scheme scheme(sk, ring);
scheme.addLeftRotKeys(sk);
scheme.addRightRotKeys(sk);
scheme.addBootKey(sk, params.bBits, params.logq + params.logI);

// Simple Constructor for secureML class
// This will just pass some address of scheme, params, and pool
SecureML::ML secureML(scheme, params, pool, sk);

// Encrypt Training Data 
// Those ciphertexts will be save in encData folder
secureML.EncryptzData(zData, factorNum, sampleNum);

// Training
// Plaintext training is in encrypted trainig for debug
double* pwData = new double[params.factorNum]();
Ciphertext* encWData = new Ciphertext[params.cnum];
secureML.Training(encWData, factorNum, sampleNum, pwData, zData);
```
