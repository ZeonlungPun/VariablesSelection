#include <iostream>
#include <vector>
#include <string>
#include <armadillo> 

using namespace std;
using namespace arma;

class GradientLearning{
public:
        mat x;
        mat y;
        float Sigma;
        mat dist_mat;
        float eps;
        float lambd;

        GradientLearning(mat xx, mat yy,float eps=0.05, float lambd=0.005)
        {
            this->x=xx;
            this->y=yy;
            this->eps=eps;
            this->lambd=lambd;
        }

        mat GaussianKernel(int sigma=8)
        {
            //accumulated sum per row nxp
            mat temp=cumsum( this->x % this->x,1);
            // get the last column : nx1
            mat last_col=temp.col(temp.n_cols-1);
            
            // Expand the nx1 matrix to a nxn matrix
            mat last_col1= repmat(last_col,1,temp.n_rows);

            // Expand the 1xn matrix to a nxn matrix
            mat last_col2= repmat(last_col.t(),temp.n_rows,1);
            
            // get the distance norm
            mat temp2=last_col1+last_col2 -2* this->x * this->x.t();
            mat result=exp(-temp2/(2*pow(sigma,2)));

            return result;
        }

        float ComputeWeightVariance()
        {
             //accumulated sum per row nxp
            mat temp=cumsum( this->x % this->x,1);
            // get the last column : nx1
            mat last_col=temp.col(temp.n_cols-1);
            
            // Expand the nx1 matrix to a nxn matrix
            mat last_col1= repmat(last_col,1,temp.n_rows);

            // Expand the 1xn matrix to a nxn matrix
            mat last_col2= repmat(last_col.t(),temp.n_rows,1);
            
            // get the distance norm
            mat dist_mat=last_col1+last_col2 -2* this->x * this->x.t();
            float med=median(median(dist_mat,1));
            this->Sigma = med;
            this->dist_mat=dist_mat;
            return med;
        }

        mat GradientLearningMain(string kernel_type="Gaussian")
        {
        // the main function for gradient learning variable selection algorithm
        // :param kernel_type: "Gaussian" Kernel or   "linear" kernel
        // :return: nrm: the RKHS norm for each dimension ; F : the gradient evaluated at each sample again a p by m matrix
            //get the data sample
            int m=this->x.n_rows;
            //get the feature num
            int p=this->x.n_cols;

            this->ComputeWeightVariance();
            //computes the weight matrix of different samples mxm
            mat w=(1/ this->Sigma * pow(2*datum::pi,0.5) )*exp( -this->dist_mat/ (2*pow(this->Sigma,2)) );

            //get the kernel matrix
            mat kernel;
            if (kernel_type=="Gaussian")
            {
                kernel=this->GaussianKernel();
            }
            else
            {
                kernel=this->x * this->x.t();
            }

            //constructs the matrix of differences between all m samples
            mat xm, Mx;
            // get the last sample of x : 1xp --> px1
            xm=this->x.row(m-1).t();
            //xm: px1--> pxm
            xm=repmat(xm,1,m);
            Mx=x.t()-xm;
        
            //SVD decompose
            mat V,UT;
            vec S;
            svd(V,S,UT,Mx);
            //inverse accumulate (begin from smallest)
            S=reverse(S);
            // cumulate
            vec cum_S=cumsum(S);
            
            //find the drop out index according to the ratio of  accumulate singular value
            vec judge= conv_to<vec>::from((cum_S/cum_S(S.n_rows-1)) < this->eps ) ;
            int cut_index = max(find(judge==1));

            //get the remain number
            int s=p-cut_index-1;
            
            //projects of the paired differences into the subspace of the s eigenfunctions
            mat t(s,m,fill::zeros);
            for (int j=0;j<m;j++)
            {
                t.col(j)=S.subvec(0,s-1)% UT.col(j).subvec(0,s-1); 
            }
            

            //initialize the transient matrix
            mat ktilde(m*s,m*s,fill::zeros);
            vec ytilde(m*s,fill::zeros);

            //computes the Ktilde matrix and the vector script Y
            for (int i=0;i<m;i++)
            {
                mat Bmat(s,s,fill::zeros);
                mat yi(s,1,fill::zeros);
                for (int j=0;j<m;j++)
                {
                    mat temp1=(t.col(j)-t.col(i)) * (t.col(j)-t.col(i)).t();
                    mat temp2= w(i,j)* ones<mat>( temp1.n_rows,temp1.n_cols );

                    Bmat+=temp1 % temp2;
                    
                    
                    mat temp3= t.col(j)- t.col(i) ;
                    mat temp4= w(i,j)*(this->y.row(j)-this->y.row(i) );
                    temp4=repmat(temp4, temp3.n_rows,temp3.n_cols);
                    yi+= temp3 % temp4;
                    
                }
                ytilde.subvec(i*s,(i+1)*s-1)=yi;
                for (int k=0;k<m;k++)
                {
                    ktilde.submat(i * s, k * s, (i + 1) * s - 1, (k + 1) * s - 1) = kernel(i,k)*Bmat;
                }
            }
            
            //solves the linear system for coefficients c
            mat I=eye<mat>(m*s,m*s);
            mat denominator = pow(m,2)*this->lambd * I +ktilde;
            mat c=inv(denominator)*ytilde;

            //uwraps the coefficients into a vector for each sample
            mat Cmat(p,m,fill::zeros);
            for (int i=0;i<m;i++)
            {
                vec vec1(p,fill::zeros);
                for (int l=0; l<s;l++)
                {
                    mat temp4=V.col(l);
                    mat temp5=c.row(i*s+l);
                    temp5=repmat(temp5,temp4.n_rows,temp4.n_cols);

                    vec1+=temp4%temp5;
                }
                Cmat.col(i)=vec1;
            }
            

            //computes the gradient for each sample
            mat F=Cmat* kernel;

            mat nrm(p,1,fill::zeros);
            for (int i=0;i<p;i++)
            {
                nrm.row(i)=Cmat.row(i)*kernel * Cmat.row(i).t();
            }

            cout<<nrm<<endl;
           


            return nrm;

        }

};
 


int main() {
    // create input and out data
    int n=200;
    int p=50;
    float xita=0.65;
    mat w=randn<mat>(n,p);
    mat u=randn<mat>(n,p);
    mat x,y;
    //create input
    x=(w+xita*u)/(1+xita);
    //create output
    y=(2*x.col(0)-1)%(2*x.col(1)-1);
    

    GradientLearning gl(x,y);
    mat nrm=gl.GradientLearningMain();

    // cout<<kernel.n_rows<<endl;
    // cout<<kernel.n_cols<<endl;
    // cout<<sigma<<endl;


    



 
    
    return 0;
}