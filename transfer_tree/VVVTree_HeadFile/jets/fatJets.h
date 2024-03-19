#ifndef _fatJets_
#define _fatJets_

#include <TROOT.h>
#include <TChain.h>
#include <TFile.h>
#include "TH2.h"
#include "TStyle.h"
#include "TCanvas.h"
#include "TMath.h"
#include "TLorentzVector.h"
//#include "DataFormats/Math/interface/deltaR.h"

#include <iostream>
#include <fstream>
#include <unordered_map>
#include <map>

#include <cstring>


#include <algorithm>
#include <numeric>
#include<vector>
#include <iostream>

struct FatJet_P //define the necessary struct used by 
{
    vector<float>* v_FatJet_pt_;
    vector<float>* v_FatJet_eta_;
    vector<float>* v_FatJet_phi_;
    vector<float>* v_FatJet_msoftdrop_;
    vector<int>* v_FatJet_jetId_;
    vector<float>* v_FatJet_tau1_;
    vector<float>* v_FatJet_tau2_;
    vector<float>* v_FatJet_tau3_;
    vector<float>* v_FatJet_tau4_;

    vector<float>* v_FatJet_msoftdrop_raw_;
    vector<float>* v_FatJet_msoftdrop_nom_;
    vector<float>* v_FatJet_msoftdrop_corr_JMR_;
    vector<float>* v_FatJet_msoftdrop_corr_JMS_;
    vector<float>* v_FatJet_msoftdrop_corr_PUPPI_;

// Add JES, JER up and down branch on 26/2/2024
    vector<float>* v_FatJet_msoftdrop_jesTotalUp_;
    vector<float>* v_FatJet_msoftdrop_jesTotalDown_;
    vector<float>* v_FatJet_msoftdrop_jerUp_;
    vector<float>* v_FatJet_msoftdrop_jerDown_;


// Add JMS, JMR up and down branch on 1/3/2024
    vector<float>* v_FatJet_msoftdrop_jmsUp_;
    vector<float>* v_FatJet_msoftdrop_jmsDown_;
    vector<float>* v_FatJet_msoftdrop_jmrUp_;
    vector<float>* v_FatJet_msoftdrop_jmrDown_;

    vector<float>* v_FatJet_pt_nom_;
    
    vector<float>* v_FatJet_inclParTMDV2_probHbb_;
    vector<float>* v_FatJet_inclParTMDV2_probHbc_;
    vector<float>* v_FatJet_inclParTMDV2_probHbs_;
    vector<float>* v_FatJet_inclParTMDV2_probHcc_;
    vector<float>* v_FatJet_inclParTMDV2_probHcs_;
    vector<float>* v_FatJet_inclParTMDV2_probHss_;
    vector<float>* v_FatJet_inclParTMDV2_probHqq_;
    vector<float>* v_FatJet_inclParTMDV2_probQCDb_;
    vector<float>* v_FatJet_inclParTMDV2_probQCDbb_;
    vector<float>* v_FatJet_inclParTMDV2_probQCDc_;
    vector<float>* v_FatJet_inclParTMDV2_probQCDcc_;
    vector<float>* v_FatJet_inclParTMDV2_probQCDothers_;

    //Start Hidden.

} ;




class FatJet_Collection {
    public : 
    FatJet_Collection();
    FatJet_Collection(FatJet_P FatJet_, int NMAXFatJet, int filterMode);
    size_t size();
    float Get(string order, string variable, int index); //defined in this file.

    private :
    // original branches
    vector<float>* v_FatJet_pt_;
    vector<float>* v_FatJet_eta_;
    vector<float>* v_FatJet_phi_;
    vector<float>* v_FatJet_msoftdrop_;
    vector<int>*   v_FatJet_jetId_;

    vector<float>* v_FatJet_tau1_;
    vector<float>* v_FatJet_tau2_;
    vector<float>* v_FatJet_tau3_;
    vector<float>* v_FatJet_tau4_;

    vector<float>* v_FatJet_msoftdrop_raw_;
    vector<float>* v_FatJet_msoftdrop_nom_;
    vector<float>* v_FatJet_msoftdrop_corr_JMR_;
    vector<float>* v_FatJet_msoftdrop_corr_JMS_;
    vector<float>* v_FatJet_msoftdrop_corr_PUPPI_;
    vector<float>* v_FatJet_pt_nom_;

// Add JES, JER up and down branch on 26/2/2024
    vector<float>* v_FatJet_msoftdrop_jesTotalUp_;
    vector<float>* v_FatJet_msoftdrop_jesTotalDown_;
    vector<float>* v_FatJet_msoftdrop_jerUp_;
    vector<float>* v_FatJet_msoftdrop_jerDown_;

// Add JES, JER up and down branch on 26/2/2024
    vector<float>* v_FatJet_msoftdrop_jmsUp_;
    vector<float>* v_FatJet_msoftdrop_jmsDown_;
    vector<float>* v_FatJet_msoftdrop_jmrUp_;
    vector<float>* v_FatJet_msoftdrop_jmrDown_;

        //Tagger V2

    vector<float>* v_FatJet_inclParTMDV2_probHbb_;
    vector<float>* v_FatJet_inclParTMDV2_probHbc_;
    vector<float>* v_FatJet_inclParTMDV2_probHbs_;
    vector<float>* v_FatJet_inclParTMDV2_probHcc_;
    vector<float>* v_FatJet_inclParTMDV2_probHcs_;
    vector<float>* v_FatJet_inclParTMDV2_probHss_;
    vector<float>* v_FatJet_inclParTMDV2_probHqq_;
    vector<float>* v_FatJet_inclParTMDV2_probQCDb_;
    vector<float>* v_FatJet_inclParTMDV2_probQCDbb_;
    vector<float>* v_FatJet_inclParTMDV2_probQCDc_;
    vector<float>* v_FatJet_inclParTMDV2_probQCDcc_;
    vector<float>* v_FatJet_inclParTMDV2_probQCDothers_;

    //Start Hidden.


    // filtered branches
    vector<float> FatJet_pt;
    vector<float> FatJet_eta;
    vector<float> FatJet_phi;
    vector<float> FatJet_msoftdrop;
    vector<int>   FatJet_jetId;

    vector<float> FatJet_tau1;
    vector<float> FatJet_tau2;
    vector<float> FatJet_tau3;
    vector<float> FatJet_tau4;
    vector<float> FatJet_msoftdrop_raw;
    vector<float> FatJet_msoftdrop_nom;
    vector<float> FatJet_msoftdrop_corr_JMR;
    vector<float> FatJet_msoftdrop_corr_JMS;
    vector<float> FatJet_msoftdrop_corr_PUPPI;

    // Add JES, JER up and down branch on 26/2/2024
    vector<float> FatJet_msoftdrop_jesTotalUp;
    vector<float> FatJet_msoftdrop_jesTotalDown;
    vector<float> FatJet_msoftdrop_jerUp;
    vector<float> FatJet_msoftdrop_jerDown;

    //JMS/JMR
    vector<float> FatJet_msoftdrop_jmsUp;
    vector<float> FatJet_msoftdrop_jmsDown;
    vector<float> FatJet_msoftdrop_jmrUp;
    vector<float> FatJet_msoftdrop_jmrDown;


    vector<float> FatJet_pt_nom;


        //Tagger V2

    vector<float> FatJet_inclParTMDV2_probHbb;
    vector<float> FatJet_inclParTMDV2_probHbc;
    vector<float> FatJet_inclParTMDV2_probHbs;
    vector<float> FatJet_inclParTMDV2_probHcc;
    vector<float> FatJet_inclParTMDV2_probHcs;
    vector<float> FatJet_inclParTMDV2_probHss;
    vector<float> FatJet_inclParTMDV2_probHqq;
    vector<float> FatJet_inclParTMDV2_probQCDb;
    vector<float> FatJet_inclParTMDV2_probQCDbb;
    vector<float> FatJet_inclParTMDV2_probQCDc;
    vector<float> FatJet_inclParTMDV2_probQCDcc;
    vector<float> FatJet_inclParTMDV2_probQCDothers;

    //Start Hidden.



    // branches map
    std::map<std::string, vector<int>>   FatJet_order; //map to store different order information.
    std::map<std::string, vector<float> > FatJet_Branches;

    int NMAXFatJet;
    int filterMode;

    void brances_map();

    void Filter();
    void Filter_1();
    void Filter_2();

    void Order(int);
    void Order_1();
    void Order_2();
    void Order_3();
    void Order_4();
    void Order_5();
    void Order_6();
    void Order_7();

    void brances_Added();
     // See the instruction for template <typename T>
    template <typename T>
    void Out(vector<T>,vector<T> &);

    template <typename T>
    std::vector<int> sort_indexes(std::vector<T> v);

};

FatJet_Collection::FatJet_Collection(){
    NMAXFatJet = 0  ; 
    filterMode = -1 ;
}

//Need to read some FatJet_ from input.
FatJet_Collection::FatJet_Collection(FatJet_P FatJet_ , int NMAXFatJet_, int filterMode_){
    v_FatJet_pt_                             = FatJet_.v_FatJet_pt_;
    v_FatJet_eta_                            = FatJet_.v_FatJet_eta_;
    v_FatJet_phi_                            = FatJet_.v_FatJet_phi_;
    v_FatJet_msoftdrop_                      = FatJet_.v_FatJet_msoftdrop_;
    v_FatJet_jetId_                          = FatJet_.v_FatJet_jetId_;


    v_FatJet_tau1_                           = FatJet_.v_FatJet_tau1_;
    v_FatJet_tau2_                           = FatJet_.v_FatJet_tau2_;
    v_FatJet_tau3_                           = FatJet_.v_FatJet_tau3_;
    v_FatJet_tau4_                           = FatJet_.v_FatJet_tau4_;
    v_FatJet_msoftdrop_raw_                  = FatJet_.v_FatJet_msoftdrop_raw_;
    v_FatJet_msoftdrop_nom_                  = FatJet_.v_FatJet_msoftdrop_nom_;
    v_FatJet_msoftdrop_corr_JMR_             = FatJet_.v_FatJet_msoftdrop_corr_JMR_;
    v_FatJet_msoftdrop_corr_JMS_             = FatJet_.v_FatJet_msoftdrop_corr_JMS_;
    v_FatJet_msoftdrop_corr_PUPPI_           = FatJet_.v_FatJet_msoftdrop_corr_PUPPI_;
   
// Add JES, JER up and down branch on 26/2/2024

    v_FatJet_msoftdrop_jesTotalUp_           = FatJet_.v_FatJet_msoftdrop_jesTotalUp_;
    v_FatJet_msoftdrop_jesTotalDown_         = FatJet_.v_FatJet_msoftdrop_jesTotalDown_;
    v_FatJet_msoftdrop_jerUp_                = FatJet_.v_FatJet_msoftdrop_jerUp_;
    v_FatJet_msoftdrop_jerDown_              = FatJet_.v_FatJet_msoftdrop_jerDown_;

// Add JES, JER up and down branch on 26/2/2024

    v_FatJet_msoftdrop_jmsUp_                = FatJet_.v_FatJet_msoftdrop_jmsUp_;
    v_FatJet_msoftdrop_jmsDown_              = FatJet_.v_FatJet_msoftdrop_jmsDown_;
    v_FatJet_msoftdrop_jmrUp_                = FatJet_.v_FatJet_msoftdrop_jmrUp_;
    v_FatJet_msoftdrop_jmrDown_              = FatJet_.v_FatJet_msoftdrop_jmrDown_;

    v_FatJet_pt_nom_                         = FatJet_.v_FatJet_pt_nom_;

    v_FatJet_inclParTMDV2_probHbb_           = FatJet_.v_FatJet_inclParTMDV2_probHbb_;
    v_FatJet_inclParTMDV2_probHbc_           = FatJet_.v_FatJet_inclParTMDV2_probHbc_;
    v_FatJet_inclParTMDV2_probHbs_           = FatJet_.v_FatJet_inclParTMDV2_probHbs_;
    v_FatJet_inclParTMDV2_probHcc_           = FatJet_.v_FatJet_inclParTMDV2_probHcc_;
    v_FatJet_inclParTMDV2_probHcs_           = FatJet_.v_FatJet_inclParTMDV2_probHcs_;
    v_FatJet_inclParTMDV2_probHss_           = FatJet_.v_FatJet_inclParTMDV2_probHss_;
    v_FatJet_inclParTMDV2_probHqq_           = FatJet_.v_FatJet_inclParTMDV2_probHqq_;
    v_FatJet_inclParTMDV2_probQCDb_          = FatJet_.v_FatJet_inclParTMDV2_probQCDb_;
    v_FatJet_inclParTMDV2_probQCDbb_         = FatJet_.v_FatJet_inclParTMDV2_probQCDbb_;
    v_FatJet_inclParTMDV2_probQCDc_          = FatJet_.v_FatJet_inclParTMDV2_probQCDc_;
    v_FatJet_inclParTMDV2_probQCDcc_         = FatJet_.v_FatJet_inclParTMDV2_probQCDcc_;
    v_FatJet_inclParTMDV2_probQCDothers_     = FatJet_.v_FatJet_inclParTMDV2_probQCDothers_;



    NMAXFatJet = NMAXFatJet_;
    filterMode = filterMode_;

    Filter(); //jetId eta pt requirement check again. and set the array like FatJet_inclParTMDV2_probTopbWtaumv, using the vector, push_back one by one.
    brances_map(); //establish map between string and all the array needed, already filled the array in Filter().
    brances_Added(); //define completely new branches using the raw scores or something else.
    Order(1);
    Order(2);
    // Order(3);
    // Order(4);
    Order(5);
    // Order(6);
    Order(7);
}

// return value is std::vector<int> type!
template <typename T>
std::vector<int> FatJet_Collection::sort_indexes(std::vector<T> v) {
  std::vector<int> idx(v.size()); // First, define an empty vector that has the same length as v;
  std::iota(idx.begin(), idx.end(), 0); // Then, fill it like 0,1,2,3...
  std::sort(idx.begin(), idx.end(), [v](int i1, int i2) { return v[i1] > v[i2]; }); //Finall sort the idx with the v value.
  return idx;
}

//To be noticed, here we already defined a vector named FatJet_Branches. string --> vector<float>
//We already defined the vectors like FatJet_pt.
void FatJet_Collection::brances_map(){
    FatJet_Branches["pt"]                             = FatJet_pt;
    FatJet_Branches["eta"]                            = FatJet_eta;
    FatJet_Branches["phi"]                            = FatJet_phi;
    FatJet_Branches["msoftdrop"]                      = FatJet_msoftdrop;
    FatJet_Branches["tau1"]                           = FatJet_tau1;
    FatJet_Branches["tau2"]                           = FatJet_tau2;
    FatJet_Branches["tau3"]                           = FatJet_tau3;
    FatJet_Branches["tau4"]                           = FatJet_tau4;
    FatJet_Branches["msoftdrop_raw"]                  = FatJet_msoftdrop_raw;
    FatJet_Branches["msoftdrop_nom"]                  = FatJet_msoftdrop_nom;
    FatJet_Branches["msoftdrop_corr_JMR"]             = FatJet_msoftdrop_corr_JMR;
    FatJet_Branches["msoftdrop_corr_JMS"]             = FatJet_msoftdrop_corr_JMS;
    FatJet_Branches["msoftdrop_corr_PUPPI"]           = FatJet_msoftdrop_corr_PUPPI;


    // Add JES, JER up and down branch on 26/2/2024
    FatJet_Branches["msoftdrop_jesTotalUp"]            = FatJet_msoftdrop_jesTotalUp;
    FatJet_Branches["msoftdrop_jesTotalDown"]          = FatJet_msoftdrop_jesTotalDown;
    FatJet_Branches["msoftdrop_jerUp"]                 = FatJet_msoftdrop_jerUp;
    FatJet_Branches["msoftdrop_jerDown"]               = FatJet_msoftdrop_jerDown;

    // Add JMS, JMR up and down

    FatJet_Branches["msoftdrop_jmsUp"]                 = FatJet_msoftdrop_jmsUp;
    FatJet_Branches["msoftdrop_jmsDown"]               = FatJet_msoftdrop_jmsDown;
    FatJet_Branches["msoftdrop_jmrUp"]                 = FatJet_msoftdrop_jmrUp;
    FatJet_Branches["msoftdrop_jmrDown"]               = FatJet_msoftdrop_jmrDown;

    // over

    FatJet_Branches["pt_nom"]                         = FatJet_pt_nom;



    FatJet_Branches["inclParTMDV2_probHbb"]            = FatJet_inclParTMDV2_probHbb;
    FatJet_Branches["inclParTMDV2_probHbc"]            = FatJet_inclParTMDV2_probHbc;
    FatJet_Branches["inclParTMDV2_probHbs"]            = FatJet_inclParTMDV2_probHbs;
    FatJet_Branches["inclParTMDV2_probHcc"]            = FatJet_inclParTMDV2_probHcc;
    FatJet_Branches["inclParTMDV2_probHcs"]            = FatJet_inclParTMDV2_probHcs;
    FatJet_Branches["inclParTMDV2_probHss"]            = FatJet_inclParTMDV2_probHss;
    FatJet_Branches["inclParTMDV2_probHqq"]            = FatJet_inclParTMDV2_probHqq;
    FatJet_Branches["inclParTMDV2_probQCDb"]           = FatJet_inclParTMDV2_probQCDb;
    FatJet_Branches["inclParTMDV2_probQCDbb"]          = FatJet_inclParTMDV2_probQCDbb;
    FatJet_Branches["inclParTMDV2_probQCDc"]           = FatJet_inclParTMDV2_probQCDc;
    FatJet_Branches["inclParTMDV2_probQCDcc"]          = FatJet_inclParTMDV2_probQCDcc;
    FatJet_Branches["inclParTMDV2_probQCDothers"]      = FatJet_inclParTMDV2_probQCDothers;
      

}

void FatJet_Collection::brances_Added(){
    vector<float> Branch;
 
    
    for (int iFJ = 0; iFJ != int(FatJet_pt.size()); ++iFJ ){
        // float HWWMD_V2 = (
        // FatJet_inclParTMDV2_probHbc[iFJ]
        // )/( 
        // FatJet_inclParTMDV2_probHbc[iFJ] +   
        // FatJet_inclParTMDV2_probQCDbb[iFJ] +   
        // FatJet_inclParTMDV2_probQCDcc[iFJ] +   
        // FatJet_inclParTMDV2_probQCDb[iFJ] +   
        // FatJet_inclParTMDV2_probQCDc[iFJ] +   
        // FatJet_inclParTMDV2_probQCDothers[iFJ] +   
        // );

        float HWWMD_V2 = FatJet_inclParTMDV2_probHbc[iFJ];

        Branch.push_back(HWWMD_V2);
    } 
    FatJet_Branches["HWW_V2"] = Branch;
    Branch.clear();
    // General HWW tagger ordering.

        
    for (int iFJ = 0; iFJ != int(FatJet_pt.size()); ++iFJ ){
        float HWWvsQCDMD_V2 = (
        FatJet_inclParTMDV2_probHbc[iFJ]
        )/(
        FatJet_inclParTMDV2_probHbc[iFJ] + 

        FatJet_inclParTMDV2_probQCDbb[iFJ] +   
        FatJet_inclParTMDV2_probQCDcc[iFJ] +   
        FatJet_inclParTMDV2_probQCDb[iFJ] +   
        FatJet_inclParTMDV2_probQCDc[iFJ] +   
        FatJet_inclParTMDV2_probQCDothers[iFJ]  
        );

        Branch.push_back(HWWvsQCDMD_V2);
        //HWWMD_v2 could change, just a temperary variable!
    } //HWWvs.QCD tagger ordering.
    
    FatJet_Branches["HWWvsQCD_V2"] = Branch;
    Branch.clear();
    //Test part1
    // for (int iBJ = 0; iBJ < FatJet_Branches["HWW_V2"].size(); iBJ++){
    //     cout << "Now the HWW_V2 branch" << iBJ << "is" << FatJet_Branches["HWW_V2"].at(iBJ) << endl;
    // }

    //Test part2
    // for (int iBJ = 0; iBJ < FatJet_Branches["HWW_V2"].size(); iBJ++){
    //     cout << "After the HWW_V2 branch" << iBJ << "is" << FatJet_Branches["HWW_V2"].at(iBJ) << endl;
    // }

    //New tagger.


    for (int iFJ = 0; iFJ != int(FatJet_pt_nom.size()); ++iFJ ){
        float out = -99 ; 
        if( FatJet_pt_nom[iFJ] > 200 ){
            out = FatJet_msoftdrop_nom[iFJ]/FatJet_msoftdrop_corr_PUPPI[iFJ];
        }
        Branch.push_back(out);
    }
    //FatJet_msoftdrop_corr_PUPPI is correction factor of PUPPI, 
    FatJet_Branches["msoftdrop_nom_noJWS"] = Branch;
    Branch.clear();


}

void FatJet_Collection::Filter(){
    if( filterMode == 1 ){
        Filter_1();
    }
    if( filterMode == 2 ){
        Filter_2();
    }
}

void FatJet_Collection::Filter_1(){
    // jetid , eta requirement
    // https://twiki.cern.ch/twiki/bin/view/CMS/JetID
    // https://twiki.cern.ch/twiki/bin/view/CMS/JetID13TeVUL
    // Please note: For AK8 jets, the corresponding (CHS or PUPPI) AK4 jet ID should be used.
    // For 2016 samples : jetId==3 means: pass loose and tight ID, fail tightLepVeto, jetId==7 means: pass loose, tight, tightLepVeto ID.
    // For 2017 and 2018 samples : jetId==2 means: pass tight ID, fail tightLepVeto, jetId==6 means: pass tight and tightLepVeto ID.
    
    //There are different jetIds, so that's why we have two different files in 2016 and 2017.

    for (size_t iFJ = 0; iFJ != v_FatJet_pt_->size(); ++iFJ ){

        int   iFatJet_jetId  = v_FatJet_jetId_->at(iFJ) ; if( iFatJet_jetId&2 != 2 ){ continue; }
        float iFatJet_jeteta = v_FatJet_eta_->at(iFJ)   ; if( fabs(iFatJet_jeteta) > 2.4 ){ continue; }
        float iFatJet_jetpt  = v_FatJet_pt_->at(iFJ)    ; if( iFatJet_jetpt < 200 ){ continue; }

        FatJet_pt.push_back(v_FatJet_pt_->at(iFJ));
        FatJet_eta.push_back(v_FatJet_eta_->at(iFJ));
        FatJet_phi.push_back(v_FatJet_phi_->at(iFJ));
        FatJet_msoftdrop.push_back(v_FatJet_msoftdrop_->at(iFJ));
        FatJet_jetId.push_back(v_FatJet_jetId_->at(iFJ));

        FatJet_tau1.push_back(v_FatJet_tau1_->at(iFJ));
        FatJet_tau2.push_back(v_FatJet_tau2_->at(iFJ));
        FatJet_tau3.push_back(v_FatJet_tau3_->at(iFJ));
        FatJet_tau4.push_back(v_FatJet_tau4_->at(iFJ));

        //V2
        FatJet_inclParTMDV2_probHbb.push_back(v_FatJet_inclParTMDV2_probHbb_->at(iFJ));
        FatJet_inclParTMDV2_probHbc.push_back(v_FatJet_inclParTMDV2_probHbc_->at(iFJ));
        FatJet_inclParTMDV2_probHbs.push_back(v_FatJet_inclParTMDV2_probHbs_->at(iFJ));
        FatJet_inclParTMDV2_probHcc.push_back(v_FatJet_inclParTMDV2_probHcc_->at(iFJ));
        FatJet_inclParTMDV2_probHcs.push_back(v_FatJet_inclParTMDV2_probHcs_->at(iFJ));
        FatJet_inclParTMDV2_probHss.push_back(v_FatJet_inclParTMDV2_probHss_->at(iFJ));
        FatJet_inclParTMDV2_probHqq.push_back(v_FatJet_inclParTMDV2_probHqq_->at(iFJ));
        FatJet_inclParTMDV2_probQCDb.push_back(v_FatJet_inclParTMDV2_probQCDb_->at(iFJ));
        FatJet_inclParTMDV2_probQCDbb.push_back(v_FatJet_inclParTMDV2_probQCDbb_->at(iFJ));
        FatJet_inclParTMDV2_probQCDc.push_back(v_FatJet_inclParTMDV2_probQCDc_->at(iFJ));
        FatJet_inclParTMDV2_probQCDcc.push_back(v_FatJet_inclParTMDV2_probQCDcc_->at(iFJ));
        FatJet_inclParTMDV2_probQCDothers.push_back(v_FatJet_inclParTMDV2_probQCDothers_->at(iFJ));

        if( v_FatJet_msoftdrop_raw_->size() > iFJ ){
            FatJet_msoftdrop_raw.push_back(v_FatJet_msoftdrop_raw_->at(iFJ));
        }
        else{
            FatJet_msoftdrop_raw.push_back(-99.);
        }

        if( v_FatJet_msoftdrop_nom_->size() > iFJ ){
            FatJet_msoftdrop_nom.push_back(v_FatJet_msoftdrop_nom_->at(iFJ));
        }
        else{
            FatJet_msoftdrop_nom.push_back(-99.);
        }

        if( v_FatJet_msoftdrop_corr_JMR_->size() > iFJ ){
            FatJet_msoftdrop_corr_JMR.push_back(v_FatJet_msoftdrop_corr_JMR_->at(iFJ));
        }
        else{
            FatJet_msoftdrop_corr_JMR.push_back(-99.);
        }

        if( v_FatJet_msoftdrop_corr_JMS_->size() > iFJ ){
            FatJet_msoftdrop_corr_JMS.push_back(v_FatJet_msoftdrop_corr_JMS_->at(iFJ));
        }
        else{
            FatJet_msoftdrop_corr_JMS.push_back(-99.);
        }

        if( v_FatJet_msoftdrop_corr_PUPPI_->size() > iFJ ){
            FatJet_msoftdrop_corr_PUPPI.push_back(v_FatJet_msoftdrop_corr_PUPPI_->at(iFJ));
        }
        else{
            FatJet_msoftdrop_corr_PUPPI.push_back(-99.);
        }

        // Add JES, JER up and down branch on 26/2/2024
        
        if( v_FatJet_msoftdrop_jesTotalUp_->size() > iFJ ){
            FatJet_msoftdrop_jesTotalUp.push_back(v_FatJet_msoftdrop_jesTotalUp_->at(iFJ));
        }
        else{
            FatJet_msoftdrop_jesTotalUp.push_back(-99.);
        }

        if( v_FatJet_msoftdrop_jesTotalDown_->size() > iFJ ){
            FatJet_msoftdrop_jesTotalDown.push_back(v_FatJet_msoftdrop_jesTotalDown_->at(iFJ));
        }
        else{
            FatJet_msoftdrop_jesTotalDown.push_back(-99.);
        }

        if( v_FatJet_msoftdrop_jerUp_->size() > iFJ ){
            FatJet_msoftdrop_jerUp.push_back(v_FatJet_msoftdrop_jerUp_->at(iFJ));
        }
        else{
            FatJet_msoftdrop_jerUp.push_back(-99.);
        }

        if( v_FatJet_msoftdrop_jerDown_->size() > iFJ ){
            FatJet_msoftdrop_jerDown.push_back(v_FatJet_msoftdrop_jerDown_->at(iFJ));
        }
        else{
            FatJet_msoftdrop_jerDown.push_back(-99.);
        }



        // Add JMS, JMR up and down branch on 1/3/2024
        
        if( v_FatJet_msoftdrop_jmsUp_->size() > iFJ ){
            FatJet_msoftdrop_jmsUp.push_back(v_FatJet_msoftdrop_jmsUp_->at(iFJ));
        }
        else{
            FatJet_msoftdrop_jmsUp.push_back(-99.);
        }

        if( v_FatJet_msoftdrop_jmsDown_->size() > iFJ ){
            FatJet_msoftdrop_jmsDown.push_back(v_FatJet_msoftdrop_jmsDown_->at(iFJ));
        }
        else{
            FatJet_msoftdrop_jmsDown.push_back(-99.);
        }

        if( v_FatJet_msoftdrop_jmrUp_->size() > iFJ ){
            FatJet_msoftdrop_jmrUp.push_back(v_FatJet_msoftdrop_jmrUp_->at(iFJ));
        }
        else{
            FatJet_msoftdrop_jmrUp.push_back(-99.);
        }

        if( v_FatJet_msoftdrop_jmrDown_->size() > iFJ ){
            FatJet_msoftdrop_jmrDown.push_back(v_FatJet_msoftdrop_jmrDown_->at(iFJ));
        }
        else{
            FatJet_msoftdrop_jmrDown.push_back(-99.);
        }

        // over


        if( v_FatJet_pt_nom_->size() > iFJ ){
            FatJet_pt_nom.push_back(v_FatJet_pt_nom_->at(iFJ));
        }
        else{
            FatJet_pt_nom.push_back(-99.);
        }

    }
}

void FatJet_Collection::Filter_2(){
    // jetid , eta requirement
    // https://twiki.cern.ch/twiki/bin/view/CMS/JetID
    // https://twiki.cern.ch/twiki/bin/view/CMS/JetID13TeVUL
    // Please note: For AK8 jets, the corresponding (CHS or PUPPI) AK4 jet ID should be used.
    // For 2016 samples : jetId==3 means: pass loose and tight ID, fail tightLepVeto, jetId==7 means: pass loose, tight, tightLepVeto ID.
    // For 2017 and 2018 samples : jetId==2 means: pass tight ID, fail tightLepVeto, jetId==6 means: pass tight and tightLepVeto ID.
    for (size_t iFJ = 0; iFJ != v_FatJet_pt_->size(); ++iFJ ){

        int   iFatJet_jetId  = v_FatJet_jetId_->at(iFJ) ; if( iFatJet_jetId&2 != 2 ){ continue; }
        float iFatJet_jeteta = v_FatJet_eta_->at(iFJ)   ; if( fabs(iFatJet_jeteta) > 2.4 ){ continue; }
        float iFatJet_jetpt  = v_FatJet_pt_nom_->at(iFJ)    ; if( iFatJet_jetpt < 200 ){ continue; }

        FatJet_pt.push_back(v_FatJet_pt_->at(iFJ));
        FatJet_eta.push_back(v_FatJet_eta_->at(iFJ));
        FatJet_phi.push_back(v_FatJet_phi_->at(iFJ));
        FatJet_msoftdrop.push_back(v_FatJet_msoftdrop_->at(iFJ));
        FatJet_jetId.push_back(v_FatJet_jetId_->at(iFJ));

        FatJet_tau1.push_back(v_FatJet_tau1_->at(iFJ));
        FatJet_tau2.push_back(v_FatJet_tau2_->at(iFJ));
        FatJet_tau3.push_back(v_FatJet_tau3_->at(iFJ));
        FatJet_tau4.push_back(v_FatJet_tau4_->at(iFJ));

        //New tagger.

        FatJet_inclParTMDV2_probHbb.push_back(v_FatJet_inclParTMDV2_probHbb_->at(iFJ));
        FatJet_inclParTMDV2_probHbc.push_back(v_FatJet_inclParTMDV2_probHbc_->at(iFJ));
        FatJet_inclParTMDV2_probHbs.push_back(v_FatJet_inclParTMDV2_probHbs_->at(iFJ));
        FatJet_inclParTMDV2_probHcc.push_back(v_FatJet_inclParTMDV2_probHcc_->at(iFJ));
        FatJet_inclParTMDV2_probHcs.push_back(v_FatJet_inclParTMDV2_probHcs_->at(iFJ));
        FatJet_inclParTMDV2_probHss.push_back(v_FatJet_inclParTMDV2_probHss_->at(iFJ));
        FatJet_inclParTMDV2_probHqq.push_back(v_FatJet_inclParTMDV2_probHqq_->at(iFJ));
        FatJet_inclParTMDV2_probQCDb.push_back(v_FatJet_inclParTMDV2_probQCDb_->at(iFJ));
        FatJet_inclParTMDV2_probQCDbb.push_back(v_FatJet_inclParTMDV2_probQCDbb_->at(iFJ));
        FatJet_inclParTMDV2_probQCDc.push_back(v_FatJet_inclParTMDV2_probQCDc_->at(iFJ));
        FatJet_inclParTMDV2_probQCDcc.push_back(v_FatJet_inclParTMDV2_probQCDcc_->at(iFJ));
        FatJet_inclParTMDV2_probQCDothers.push_back(v_FatJet_inclParTMDV2_probQCDothers_->at(iFJ));

        if( v_FatJet_msoftdrop_raw_->size() > iFJ ){
            FatJet_msoftdrop_raw.push_back(v_FatJet_msoftdrop_raw_->at(iFJ));
        }
        else{
            FatJet_msoftdrop_raw.push_back(-99.);
        }

        if( v_FatJet_msoftdrop_nom_->size() > iFJ ){
            FatJet_msoftdrop_nom.push_back(v_FatJet_msoftdrop_nom_->at(iFJ));
        }
        else{
            FatJet_msoftdrop_nom.push_back(-99.);
        }

        if( v_FatJet_msoftdrop_corr_JMR_->size() > iFJ ){
            FatJet_msoftdrop_corr_JMR.push_back(v_FatJet_msoftdrop_corr_JMR_->at(iFJ));
        }
        else{
            FatJet_msoftdrop_corr_JMR.push_back(-99.);
        }

        if( v_FatJet_msoftdrop_corr_JMS_->size() > iFJ ){
            FatJet_msoftdrop_corr_JMS.push_back(v_FatJet_msoftdrop_corr_JMS_->at(iFJ));
        }
        else{
            FatJet_msoftdrop_corr_JMS.push_back(-99.);
        }

        if( v_FatJet_msoftdrop_corr_PUPPI_->size() > iFJ ){
            FatJet_msoftdrop_corr_PUPPI.push_back(v_FatJet_msoftdrop_corr_PUPPI_->at(iFJ));
        }
        else{
            FatJet_msoftdrop_corr_PUPPI.push_back(-99.);
        }


        // Add JES, JER up and down branch on 26/2/2024
        
        if( v_FatJet_msoftdrop_jesTotalUp_->size() > iFJ ){
            FatJet_msoftdrop_jesTotalUp.push_back(v_FatJet_msoftdrop_jesTotalUp_->at(iFJ));
        }
        else{
            FatJet_msoftdrop_jesTotalUp.push_back(-99.);
        }

        if( v_FatJet_msoftdrop_jesTotalDown_->size() > iFJ ){
            FatJet_msoftdrop_jesTotalDown.push_back(v_FatJet_msoftdrop_jesTotalDown_->at(iFJ));
        }
        else{
            FatJet_msoftdrop_jesTotalDown.push_back(-99.);
        }

        if( v_FatJet_msoftdrop_jerUp_->size() > iFJ ){
            FatJet_msoftdrop_jerUp.push_back(v_FatJet_msoftdrop_jerUp_->at(iFJ));
        }
        else{
            FatJet_msoftdrop_jerUp.push_back(-99.);
        }

        if( v_FatJet_msoftdrop_jerDown_->size() > iFJ ){
            FatJet_msoftdrop_jerDown.push_back(v_FatJet_msoftdrop_jerDown_->at(iFJ));
        }
        else{
            FatJet_msoftdrop_jerDown.push_back(-99.);
        }


        if( v_FatJet_pt_nom_->size() > iFJ ){
            FatJet_pt_nom.push_back(v_FatJet_pt_nom_->at(iFJ));
        }
        else{
            FatJet_pt_nom.push_back(-99.);
        }


        // start JMS/JMR
             
        if( v_FatJet_msoftdrop_jmsUp_->size() > iFJ ){
            FatJet_msoftdrop_jmsUp.push_back(v_FatJet_msoftdrop_jmsUp_->at(iFJ));
        }
        else{
            FatJet_msoftdrop_jmsUp.push_back(-99.);
        }

        if( v_FatJet_msoftdrop_jmsDown_->size() > iFJ ){
            FatJet_msoftdrop_jmsDown.push_back(v_FatJet_msoftdrop_jmsDown_->at(iFJ));
        }
        else{
            FatJet_msoftdrop_jmsDown.push_back(-99.);
        }

        if( v_FatJet_msoftdrop_jmrUp_->size() > iFJ ){
            FatJet_msoftdrop_jmrUp.push_back(v_FatJet_msoftdrop_jmrUp_->at(iFJ));
        }
        else{
            FatJet_msoftdrop_jmrUp.push_back(-99.);
        }

        if( v_FatJet_msoftdrop_jmrDown_->size() > iFJ ){
            FatJet_msoftdrop_jmrDown.push_back(v_FatJet_msoftdrop_jmrDown_->at(iFJ));
        }
        else{
            FatJet_msoftdrop_jmrDown.push_back(-99.);
        }

        // over   

    }
}

size_t FatJet_Collection::size(){
    return FatJet_pt.size();
}

void FatJet_Collection::Order(int mode){
    if((mode == 1) && (NMAXFatJet == 3)){ Order_1(); }
    if((mode == 2) && (NMAXFatJet == 3)){ Order_2(); }
    if((mode == 3) && (NMAXFatJet == 3)){ Order_3(); }
    if((mode == 4) && (NMAXFatJet == 3)){ Order_4(); }
    if((mode == 5) && (NMAXFatJet == 3)){ Order_5(); }

    if((mode == 6) && (NMAXFatJet == 3)){ Order_6(); } //HWWMD H4q tagger order.
    if((mode == 7) && (NMAXFatJet == 3)){ Order_7(); } //HWWMD V2 tagger order.
    

}

void FatJet_Collection::Order_1(){
    vector<int> FJorder;
    if( FatJet_pt.size() == 2 ){
        FJorder.push_back(0);
        FJorder.push_back(1);
        FJorder.push_back(-99);
    }
    if( FatJet_pt.size() == 3 ){
        FJorder.push_back(0);
        FJorder.push_back(1);
        FJorder.push_back(2);
    }
    else{
        FJorder.push_back(-99);
        FJorder.push_back(-99);
        FJorder.push_back(-99);
    }
    FatJet_order["PT"] = FJorder;
}

void FatJet_Collection::Order_2(){
    vector<int> FJorder;
    vector<int> FJorder_tmp = sort_indexes(FatJet_msoftdrop);
    if( FatJet_pt.size() == 2 ){
        FJorder.push_back(FJorder_tmp[0]);
        FJorder.push_back(-99);
        FJorder.push_back(FJorder_tmp[1]);
    }
    if( FatJet_pt.size() == 3 ){
        FJorder = FJorder_tmp ;
    }
    else{
        FJorder.push_back(-99);
        FJorder.push_back(-99);
        FJorder.push_back(-99);
    }
    FatJet_order["Mass"] = FJorder;
}

void FatJet_Collection::Order_3(){
    // deep-W MD ordered
    vector<float> OrderBranch;
    OrderBranch = FatJet_Branches["deepTagMD_WvsQCD"];

    vector<int> FJorder;
    vector<int> FJorder_tmp = sort_indexes(OrderBranch);
    if( FatJet_pt.size() == 2 ){
        FJorder.push_back(FJorder_tmp[0]);
        FJorder.push_back(-99);
        FJorder.push_back(FJorder_tmp[1]);
    }
    if( FatJet_pt.size() == 3 ){
        FJorder = FJorder_tmp ;
    }
    else{
        FJorder.push_back(-99);
        FJorder.push_back(-99);
        FJorder.push_back(-99);
    }
    FatJet_order["deep-W-MD"] = FJorder;
}

void FatJet_Collection::Order_4(){
    // paricle-net W MD ordered
    vector<float> OrderBranch;
    OrderBranch = FatJet_Branches["particleNetMD_WvsQCD"];

    vector<int> FJorder;
    vector<int> FJorder_tmp = sort_indexes(OrderBranch);
    if( FatJet_pt.size() == 2 ){
        FJorder.push_back(FJorder_tmp[0]);
        FJorder.push_back(-99);
        FJorder.push_back(FJorder_tmp[1]);
    }
    if( FatJet_pt.size() == 3 ){
        FJorder = FJorder_tmp ;
    }
    else{
        FJorder.push_back(-99);
        FJorder.push_back(-99);
        FJorder.push_back(-99);
    }
    FatJet_order["PNet-W-MD"] = FJorder;
}

void FatJet_Collection::Order_5(){
    // paricle-net W MD ordered
    vector<float> OrderBranch;
    OrderBranch = FatJet_Branches["pt_nom"];

    vector<int> FJorder;
    vector<int> FJorder_tmp = sort_indexes(OrderBranch);
    if( FatJet_pt.size() == 2 ){
        FJorder.push_back(FJorder_tmp[0]);
        FJorder.push_back(FJorder_tmp[1]);
        FJorder.push_back(-99);
    }
    if( FatJet_pt.size() == 3 ){
        FJorder = FJorder_tmp ;
    }
    else{
        FJorder.push_back(-99);
        FJorder.push_back(-99);
        FJorder.push_back(-99);
    }
    FatJet_order["pt_nom"] = FJorder;
}

void FatJet_Collection::Order_6(){
    //General Higgs tagger ordered
    vector<float> OrderBranch;
    OrderBranch = FatJet_Branches["deepHWWMDV2_HallvsQCD"];

    vector<int> FJorder;
    vector<int> FJorder_tmp = sort_indexes(OrderBranch);
    if( FatJet_pt.size() == 2 ){
        FJorder.push_back(FJorder_tmp[0]);
        FJorder.push_back(-99);
        FJorder.push_back(FJorder_tmp[1]);
    }
    if( FatJet_pt.size() == 3 ){
        FJorder = FJorder_tmp ;
    }
    else{
        FJorder.push_back(-99);
        FJorder.push_back(-99);
        FJorder.push_back(-99);
    }
    FatJet_order["HWW-H4q-MD"] = FJorder;
}

void FatJet_Collection::Order_7(){
    //newest Higgs tagger ordered
    vector<float> OrderBranch;
    OrderBranch = FatJet_Branches["HWW_V2"];

    vector<int> FJorder;
    vector<int> FJorder_tmp = sort_indexes(OrderBranch);
    if( FatJet_pt.size() == 2 ){
        FJorder.push_back(FJorder_tmp[0]);
        FJorder.push_back(-99);
        FJorder.push_back(FJorder_tmp[1]);
    }
    if( FatJet_pt.size() == 3 ){
        FJorder = FJorder_tmp ;
    }
    else{
        FJorder.push_back(-99);
        FJorder.push_back(-99);
        FJorder.push_back(-99);
    }
    FatJet_order["HWW-V2-MD"] = FJorder;
    // To be noticed here, the FatJet_order is a map!!! string --> vector. just like the dict in python.
    // The function provided the FatJet_order["HWW-V2-MD"] , a vector contain jet order information like [0,1,2] or [2,1,0]
}



float FatJet_Collection::Get(string order, string variable, int index){
    int id = FatJet_order[order][index];
    if( id == -99 ){
        return -99.;
    }
    else{
        return FatJet_Branches[variable].at(id);
    }
    // To get the given order's no.index variable.
}

// template <typename T>
// void FatJet_Collection::Out(vector<T> In_branch, vector<T> & Out_branch){
    // for (size_t iFJ = 0; iFJ != FJorder.size(); ++iFJ ){ 
        // if(iFJ < In_branch.size()){ Out_branch.push_back( In_branch.at(FJorder.at(iFJ)) ); }
        // else{ Out_branch.push_back(-99); }
    // }
// }

#endif