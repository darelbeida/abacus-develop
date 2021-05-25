#include "./stress_func.h"
#include "./xc_functional.h"
#include "./myfunc.h"
#include "./xc_gga_pw.h"

//calculate the GGA stress correction in PW and LCAO
void Stress_Func::stress_gga(matrix& sigma) 
{
	timer::tick("Stress_Func","stress_gga",'F');
     
	if (xcf.igcx == 0  &&  xcf.igcc == 0)
	{
		return;
	} 
	double sigma_gradcorr[3][3];
	double* p= &sigma_gradcorr[0][0];
	for(int i=0;i<9;i++)
		*p++ = 0;

	bool igcc_is_lyp = false;
	if( xcf.igcc == 3 || xcf.igcc == 7)
	{
		igcc_is_lyp = true;
	}

	assert(NSPIN>0);
	const double fac = 1.0/ NSPIN;

	// doing FFT to get rho in G space: rhog1 
	CHR.set_rhog(CHR.rho[0], CHR.rhog[0]);
	if(NSPIN==2)//mohan fix bug 2012-05-28
	{
		CHR.set_rhog(CHR.rho[1], CHR.rhog[1]);
	}
	CHR.set_rhog(CHR.rho_core, CHR.rhog_core);
		
	double* rhotmp1;
	double* rhotmp2;
	complex<double>* rhogsum1;
	complex<double>* rhogsum2;
	Vector3<double>* gdr1;
	Vector3<double>* gdr2;
 
	rhotmp1 = new double[pw.nrxx];
	rhogsum1 = new complex<double>[pw.ngmc];
	ZEROS(rhotmp1, pw.nrxx);
	ZEROS(rhogsum1, pw.ngmc);
	for(int ir=0; ir<pw.nrxx; ir++) rhotmp1[ir] = CHR.rho[0][ir] + fac * CHR.rho_core[ir];
	for(int ig=0; ig<pw.ngmc; ig++) rhogsum1[ig] = CHR.rhog[0][ig] + fac * CHR.rhog_core[ig];
	gdr1 = new Vector3<double>[pw.nrxx];
	ZEROS(gdr1, pw.nrxx);

	GGA_PW::grad_rho( rhogsum1 , gdr1 );

	if(NSPIN==2)
	{
		rhotmp2 = new double[pw.nrxx];
		rhogsum2 = new complex<double>[pw.ngmc];
		ZEROS(rhotmp2, pw.nrxx);
		ZEROS(rhogsum2, pw.ngmc);
		for(int ir=0; ir<pw.nrxx; ir++)
		{
			rhotmp2[ir] = CHR.rho[1][ir] + fac * CHR.rho_core[ir];
		}
		for(int ig=0; ig<pw.ngmc; ig++)
		{
			rhogsum2[ig] = CHR.rhog[1][ig] + fac * CHR.rhog_core[ig];
		}
		
		gdr2 = new Vector3<double>[pw.nrxx];
		ZEROS(gdr2, pw.nrxx);

		GGA_PW::grad_rho( rhogsum2 , gdr2 );
	}
        
	const double epsr = 1.0e-6;
	const double epsg = 1.0e-10;

	double grho2a = 0.0;
	double grho2b = 0.0;
	double sx = 0.0;
	double sc = 0.0;
	double v1x = 0.0;
	double v2x = 0.0;
	double v1c = 0.0;
	double v2c = 0.0;
	double vtxcgc = 0.0;
	double etxcgc = 0.0;

	if(NSPIN==1||NSPIN==4)
	{
		double segno;
		for(int ir=0; ir<pw.nrxx; ir++)
		{
			const double arho = std::abs( rhotmp1[ir] );
			if(arho > epsr)
			{
				grho2a = gdr1[ir].norm2();
				if( grho2a > epsg )
				{
					if( rhotmp1[ir] >= 0.0 ) segno = 1.0;
					if( rhotmp1[ir] < 0.0 ) segno = -1.0;

					XC_Functional::gcxc( arho, grho2a, sx, sc, v1x, v2x, v1c, v2c);
					double tt[3];
					tt[0] = gdr1[ir].x;
					tt[1] = gdr1[ir].y;
					tt[2] = gdr1[ir].z;
					for(int l = 0;l< 3;l++)
					{
						for(int m = 0;m< l+1;m++)
						{
							sigma_gradcorr[l][m] += tt[l] * tt[m] * e2 * (v2x + v2c);
						}
					}
				}
			}
		} 
	}
	else if(NSPIN==2)
	{
		double v1cup = 0.0;
		double v1cdw = 0.0;
		double v2cup = 0.0;
		double v2cdw = 0.0;
		double v1xup = 0.0;
		double v1xdw = 0.0;
		double v2xup = 0.0;
		double v2xdw = 0.0;
		double v2cud = 0.0;
		double v2c = 0.0;
		for(int ir=0; ir<pw.nrxx; ir++)
		{
			double rh = rhotmp1[ir] + rhotmp2[ir];
			grho2a = gdr1[ir].norm2();;
			grho2b = gdr2[ir].norm2();;
			//XC_Functional::gcx_spin();
			gcx_spin(rhotmp1[ir], rhotmp2[ir], grho2a, grho2b,
				sx, v1xup, v1xdw, v2xup, v2xdw);

			if(rh > epsr)
			{
				if(igcc_is_lyp)
				{
					WARNING_QUIT("stress","igcc_is_lyp is not available now.");
				}
				else
				{
					double zeta = ( rhotmp1[ir] - rhotmp2[ir] ) / rh;
					double grh2 = (gdr1[ir]+gdr2[ir]).norm2();
					//XC_Functional::gcc_spin(rh, zeta, grh2, sc, v1cup, v1cdw, v2c);
					gcc_spin(rh, zeta, grh2, sc, v1cup, v1cdw, v2c);
					v2cup = v2c;
					v2cdw = v2c;
					v2cud = v2c;
				}
			}
			else
			{
				sc = 0.0;
				v1cup = 0.0;
				v1cdw = 0.0;
				v2c = 0.0;
				v2cup = 0.0;
				v2cdw = 0.0;
				v2cud = 0.0;
			}
			double tt1[3],tt2[3];
			{
				tt1[0] = gdr1[ir].x;
				tt1[1] = gdr1[ir].y;
				tt1[2] = gdr1[ir].z;
				tt2[0] = gdr2[ir].x;
				tt2[1] = gdr2[ir].y;
				tt2[2] = gdr2[ir].z;
			}
			for(int l = 0;l< 3;l++)
			{
			    for(int m = 0;m< l+1;m++)
				{
				//    exchange
				sigma_gradcorr [l][m] += tt1[l] * tt1[m] * e2 * v2xup + 
							tt2[l] * tt2[m] * e2 * v2xdw;
				//    correlation
				sigma_gradcorr [l][m] += ( tt1[l] * tt1[m] * v2cup + 
							tt2[l] * tt2[m] * v2cdw + 
							(tt1[l] * tt2[m] +
							tt2[l] * tt1[m] ) * v2cud ) * e2;
				}
			}
		}
	}

	for(int l = 0;l< 3;l++)
	{
		for(int m = 0;m< l;m++)
		{
			sigma_gradcorr[m][l] = sigma_gradcorr[l][m];
		}
	}
	for(int l = 0;l<3;l++)
	{
		for(int m = 0;m<3;m++)
		{
			Parallel_Reduce::reduce_double_pool( sigma_gradcorr[l][m] );
		}
	}
	
/*	p= &sigma_gradcorr[0][0];
	double* p1 = &sigmaxc[0][0];
	for(int i=0;i<9;i++){
		*p /= pw.ncxyz ;
		*p1++ += *p++;  
	}*/
	
	for(int i=0;i<3;i++)
	{
		for(int j=0;j<3;j++)
		{
			sigma(i,j) += sigma_gradcorr[i][j] / pw.ncxyz;
		}
	}

	delete[] rhotmp1;
	delete[] rhogsum1;
	delete[] gdr1;
	if(NSPIN==2)
	{
		delete[] rhotmp2;
		delete[] rhogsum2;
		delete[] gdr2;
	}
	timer::tick("Stress_Func","stress_gga");
	return;
}
