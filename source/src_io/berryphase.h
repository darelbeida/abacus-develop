#ifndef BERRYPHASE_H
#define BERRYPHASE_H

#include "../src_io/unk_overlap_pw.h"
#include "../src_io/unk_overlap_lcao.h"

class berryphase
{

public:

	berryphase();
	~berryphase();

	// mohan add 2021-02-16
	static bool berry_phase_flag;

	unkOverlap_pw pw_method;

	unkOverlap_lcao lcao_method;

	int total_string;
	vector<vector<int>> k_index;
	int nppstr;
	int direction;
	int occ_nbands;
	int GDIR;
	
	void get_occupation_bands();

	void lcao_init();

	void set_kpoints(const int direction);

	double stringPhase(int index_str, int nbands);

	void Berry_Phase(int nbands, double &pdl_elec_tot, int &mod_elec_tot);

	void Macroscopic_polarization();

	string outFormat(const double polarization, const double modulus, const Vector3<double> project);
	
};

#endif
