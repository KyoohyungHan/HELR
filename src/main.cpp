#include <iostream>
#include <stdlib.h>
#include <chrono>

#include <NTL/BasicThreadPool.h>

#include "HEAAN.h"

#include "Params.h"
#include "ML.h"
#include "functions.h"

chrono::high_resolution_clock::time_point t1, t2;

#define START() t1 = chrono::high_resolution_clock::now();
#define END() t2 = chrono::high_resolution_clock::now();
#define PRINTTIME(msg) cout << msg << " time = " << (double)chrono::duration_cast<chrono::milliseconds>(t2 - t1).count() / 1000 << " seconds" << endl;

using namespace std;
using namespace NTL;

void test(string file, string file_test, bool isFirst, long numThread, bool isEncrypted) {

	/* Construct BasicThreadPool and set the number of thread
	 * */
	BasicThreadPool pool(numThread);

	/* Read Data file and shuffle and normalize it
	 * */
	long factorNum, sampleNum;
	double** zData = SecureML::zDataFromFile(file, factorNum, sampleNum, isFirst);

	SecureML::shuffleZData(zData, factorNum, sampleNum);
	SecureML::normalizeZData(zData, factorNum, sampleNum);
	
	/* Params(long factorNum, long sampleNum, long iterNum, double alpha, long numThread)
	 * alpha : learning rate (0.001 to 1.0 values are used in general
	 * iterNum does not need to be large because we use mini-batch technique + optimized gradient decent
	 */
	SecureML::Params params(factorNum, sampleNum, 32, 1.0, pool.NumThreads());
	params.path_to_file = file;
	params.path_to_test_file = file_test;
	params.isfirst = isFirst;

	/* Key Generation for HEAANBOOT library
	 * If params.iterNum is larger than params.iterNumPerBoot, this will generate public key for bootstrapping
	 * */
	START();
	Ring ring(params.logN, params.logQBoot);
	SecretKey sk(ring);
	Scheme scheme(sk, ring);
	if(isEncrypted) {
		scheme.addLeftRotKeys(sk);
		scheme.addRightRotKeys(sk);
		scheme.addBootKey(sk, params.bBits, params.logq + params.logI);
	}
	END(); PRINTTIME("\n - KeyGen");

	/* Simple Constructor for secureML class
	 * This will just pass some address of scheme, params, and pool
	 * */
	SecureML::ML secureML(scheme, params, pool, sk);

	/* Encrypt zData using public key (this part can be faster in symmetric key version)
	 * To prepare for large number of data case, encryption of zData will be saved in encData/ folder with .txt files
	 * */
	START();
	if(isEncrypted) secureML.EncryptzData(zData, factorNum, sampleNum);
	END(); PRINTTIME(" - Encrypt");

	/* Training using zData which are saved as txt file
	 * */
	START();
	double* pwData = new double[params.factorNum]();
	Ciphertext* encWData = new Ciphertext[params.cnum];
	if(isEncrypted) secureML.Training(encWData, factorNum, sampleNum, pwData, zData);
	else secureML.plainTraining(pwData, zData, factorNum, sampleNum);
	END(); PRINTTIME("\n - Training");

	if(isEncrypted) {
		/* Decrypt encWData using secretkey sk
		 * */
		double* dwData = new double[factorNum]();
		secureML.DecryptwData(dwData, encWData, factorNum);
		
		/* Print Data Value / Pr[Y=1|X=x] for given test file using the dwData
	 	* */
		SecureML::testProbAndYval(params.path_to_test_file, dwData, params.isfirst);
		delete[] dwData;
	}
	delete[] pwData;
	delete[] encWData;
}

int main() {

	string file1("../data/MNIST_train.txt"); //> file for training
	string file2("../data/MNIST_test.txt"); //> file for testing
	bool isFirst = true; //> Y val is at the first column?

	// Un-encrypted Logistic Regression //

	// cout << "!!! Test for Thread = 1 !!!" << endl;
	// test(file, isFirst, 1, false);

	// cout << "!!! Test for Thread = 2 !!!" << endl;
	// test(file, isFirst, 2, false);

	// cout << "!!! Test for Thread = 4 !!!" << endl;
	// test(file, isFirst, 4, false);

	// cout << "!!! Test for Thread = 8 !!!" << endl;
	// test(file1, file2, isFirst, 8, false);

	// Encrypted Logistic Regression //

	// cout << "!!! Test for Thread = 1 !!!" << endl;
	// test(file, isFirst, 1, true);

	// cout << "!!! Test for Thread = 2 !!!" << endl;
	// test(file1, file2, isFirst, 2, true);

	cout << "!!! Test for Thread = 4 !!!" << endl;
	test(file1, file2, isFirst, 4, true);

	// cout << "!!! Test for Thread = 8 !!!" << endl;
	// test(file1, file2, isFirst, 8, true);

	return 0;
}



