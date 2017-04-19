#pragma once
#include"Preprocessing.h"
#include<ctime>

MatrixXd OptimizeGD(MatrixXd ins, MatrixXd answers) {

	MatrixXd inputs(ins.rows(), ins.cols() + 1);
	inputs << VectorXd::Ones(ins.rows()), ins;
	int size = inputs.cols();
	int ans = answers.cols();
	int nums = inputs.rows();
	//variables
	MatrixXd X = MatrixXd::Random(size, ans) * 10;
    MatrixXd Y = MatrixXd::Random(size, ans) * 10;
    X = (Y - X)/(2*nums);
	//MatrixXd sqPart = (X.transpose()*inputs)*X;
	//need to check when to 
	double oldErr = pow(10, 15);
	double newErr = oldErr - 1000;



	MatrixXd err;
	MatrixXd ones = MatrixXd::Ones(inputs.rows(), X.cols());//ones vector
	MatrixXd dev = (inputs - ones*ones.transpose()*inputs) / nums;//deviation matrix
	MatrixXd varcov = dev.transpose()*dev / nums;//vatiance-covariance matrix
    
    MatrixXd quad = (ones*(X.transpose()*varcov))*X;
    MatrixXd lin = (X.transpose()*inputs.transpose()).transpose();//linear part
							//MatrixXd quad = ones*((X.transpose())*(varcov)*X);//quadratic part
    double begin = clock();
	cout << "init" << endl;
	err = ((quad + lin - answers) / nums);//error - purpose function
	oldErr = newErr;
	newErr = (err.norm());
	cout << err.norm() << endl;
    MatrixXd grad = ((err.transpose()*((ones*(X.transpose()*varcov)) + inputs)).transpose()/nums + 2*X)/nums;
    cout << "here" ;
    MatrixXd Xold = X*0.9;
	double old = 0;
	double lr = 0.1;//learning rate
	while (true) {
		if (grad.squaredNorm()>0.0000000001 && (X-Xold).norm()>0.0000000001 ) {//if learning had significant effect

			old = X.norm();
			//update
            Xold = X;
            cout << "grad " << grad.norm() << endl;
			X = X - lr*grad;
			cout << "changed" << endl;
			cout << X.norm() - old << endl;
            cout << "X norm " <<endl;
            cout << X.norm() << endl;
			// lr = lr*1.01;

			cout << "inside" << endl;
			lin = (X.transpose()*inputs.transpose()).transpose();//linear part//linear part
            quad = (ones*(X.transpose()*varcov))*X;
			err = ((quad + lin - answers) / nums);//error - purpose function

			cout << err.squaredNorm() << endl;
			oldErr = newErr;
			newErr = (err.squaredNorm());
            lr = lr*0.99;
            grad = ((err.transpose()*(ones*(X.transpose()*varcov) + inputs)).transpose()/nums + 2*X)/nums;
        }
		else {
			break;
		}
	}
    cout << "time " << clock()-begin << endl;
system("pause");
	//cout << X.transpose() << endl;
	cout << err.sum() << endl;
	return X;
}

void testGD(MatrixXd ins, MatrixXd answers, MatrixXd vars) {//test
cout << "test" << endl;
	MatrixXd inputs(ins.rows(), ins.cols() + 1);
	inputs << VectorXd::Ones(ins.rows()), ins;
	int size = inputs.cols();
	int ans = answers.cols();
	int nums = inputs.rows();
     MatrixXd ones = MatrixXd::Ones(inputs.rows(), vars.cols());//ones vector
    MatrixXd dev = (inputs - ones*ones.transpose()*inputs) / nums;//deviation matrix
	MatrixXd varcov = dev.transpose()*dev / nums;//vatiance-covariance matrix
    
     
	MatrixXd lin = (vars.transpose()*inputs.transpose()).transpose();//linear part
     MatrixXd quad = (ones*(vars.transpose()*varcov))*vars;
    

	MatrixXd err = (quad + lin - answers);

	double T = 0;//true
	double F = 0;//false
    bool tr;
	for (int i = 0; i < nums; i++) {
		cout << "error " << err.row(i) << endl;
		tr = err.row(i).cwiseAbs().maxCoeff()<0.5;
		if (tr) {

			T = T+1;
		}
		else {
			F = F+1;
		}
	}

	cout << "accuracy = " << T / (T + F) << endl;

	system("pause");
}

MatrixXd Nesterov(MatrixXd ins, MatrixXd answers) {
    
	MatrixXd inputs(ins.rows(), ins.cols() + 1);
	inputs << VectorXd::Ones(ins.rows()), ins;
	int size = inputs.cols();
	int ans = answers.cols();
	int nums = inputs.rows();

	MatrixXd X = MatrixXd::Random(size, ans) * 10;
    MatrixXd Y = MatrixXd::Random(size, ans) * 10;
    X = (X-Y)/(2*nums);

	//MatrixXd sqPart = (X.transpose()*inputs)*X;

	double oldErr = pow(10, 15);
	double newErr = oldErr - 1000;



	MatrixXd err;
	MatrixXd ones = MatrixXd::Ones(inputs.rows(), X.cols());//ones vector
	MatrixXd dev = (inputs - ones*ones.transpose()*inputs) / nums;//deviation matrix
	MatrixXd varcov = dev.transpose()*dev / nums;//vatiance-covariance matrix
	MatrixXd V = MatrixXd::Zero(X.rows(), X.cols());
    MatrixXd Vin = X;
    
    MatrixXd lin = (X.transpose()*inputs.transpose()).transpose();//linear part
	 MatrixXd quad = (ones*(X.transpose()*varcov))*X;	
double begin = clock();					//MatrixXd quad = ones*((X.transpose())*(varcov)*X);//quadratic part
	cout << "init" << endl;
	err = ((quad + lin - answers) / nums);//error - purpose function
	oldErr = newErr;
	newErr = (err.norm());
	cout << err.norm() << endl;
    MatrixXd grad = ((err.transpose()*(ones*(X.transpose()*varcov) + inputs)).transpose()/nums + 2*X)/nums;
    MatrixXd Xold = X*0.9;
	double old = 0;
	double lr = 0.1;//learning rate
    double gamma = 1-lr;
	while (true) {
		if (grad.squaredNorm()>0.0000000001 && (X-Xold).norm()>0.0000000001 ) {//if learning had significant effect
            
            V = gamma*V + lr*grad;
            
			old = X.norm();
			//update
            Xold = X;
            cout << "grad " << grad.norm() << endl;
			X = X - V;
			cout << "changed" << endl;
			cout << X.norm() - old << endl;
            cout << "X norm " <<endl;
            cout << X.norm() << endl;
			// lr = lr*1.01;
            Vin = X - gamma*V;
			cout << "inside" << endl;
			lin = (Vin.transpose()*inputs.transpose()).transpose();//linear part//linear part
            MatrixXd quad = (ones*(X.transpose()*varcov))*X;
						   //MatrixXd quad = ones*((X.transpose())*(varcov)*X);//quadratic part
			err = ((quad + lin - answers)/nums);//error - purpose function

			cout << err.squaredNorm() << endl;
			oldErr = newErr;
			newErr = (err.squaredNorm());
            lr = lr*0.99;
            gamma = 1 - lr;
            grad = ((err.transpose()*(ones*(X.transpose()*varcov) + inputs)).transpose()/nums + 2*X)/nums;
		}
		else {
			break;
		}
	}
    cout << "time " << clock()-begin << endl;
system("pause");
	//cout << X.transpose() << endl;
	cout << err.sum() << endl;
	return X;
}