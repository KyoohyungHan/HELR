# HEML: Homomorphic Logistic Regression on Encrypted Data

## How to run this program?

In 'Debug' folder, just use 'make all' command (might need to type 'make clean' first if you re-construct this program).

This will construct 'HELR' file that we can run.

To test logistic regression on encrypted data, you need to activate those tests in main.cpp file.

Notice that you might need to change the path to the NTL library and HEAAN library.

NTL lib: http://www.shoup.net/ntl/

HEAAN lib: https://github.com/kimandrik/HEAAN (pre-released v1.2)

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
