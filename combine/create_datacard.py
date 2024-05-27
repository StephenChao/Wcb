"""
Creates "combine datacards" using hist.Hist templates, and
sets up data-driven QCD background estimate ('rhalphabet' method)

Adapted from
    https://github.com/rkansal47/HHbbVV/blob/main/src/HHbbVV/postprocessing/CreateDatacard.py
    https://github.com/jennetd/vbf-hbb-fits/blob/master/hbb-unblind-ewkz/make_cards.py
    https://github.com/LPC-HH/combine-hh/blob/master/create_datacard.py
    https://github.com/nsmith-/rhalphalib/blob/master/tests/test_rhalphalib.py
    https://github.com/farakiko/boostedhiggs/blob/main/combine/create_datacard.py

Author: Raghav Kansal, Yuzhe Zhao
"""

from __future__ import division, print_function

import argparse
import json
import logging
import math
import pickle as pkl
import warnings

import numpy as np
import pandas as pd
import os, sys
# sys.path.append("/ospool/cms-user/yuzhe/BoostedHWW/rhaphabetlib/CMSSW_11_3_4/src/rhalphalib")
import rhalphalib as rl
# import hist
from hist import Hist
from pathlib import Path

from typing import Dict, List, Tuple, Union
from dataclasses import dataclass, field

from collections import OrderedDict
from utils import blindBins, get_template, labels, samples, shape_to_num, sigs 

from datacardHelpers import (
    ShapeVar,
    Syst,
    add_bool_arg,
    sum_templates,
    get_effect_updown,
    get_year_updown,
    rem_neg
)

rl.ParametericSample.PreferRooParametricHist = False
logging.basicConfig(level=logging.INFO)
import argparse

warnings.filterwarnings("ignore", message="Found duplicate branch ")
pd.set_option("mode.chained_assignment", None)


CMS_PARAMS_LABEL = "CMS_Wcb_boosted"
years = ["2016APV", "2016", "2017", "2018"]

LUMI = {  # in pb^-1
    "2016": 16830.0,
    "2016APV": 19500.0,
    "2017": 41480.0,
    "2018": 59830.0,
}

jecs = {
    # "JES": "JES_jes",
    "JER": "JER",
    "Absolute": "split",
    "Absolute_year": "split",
    "BBEC1": "split",
    "BBEC1_year": "split",
    "EC2": "split",
    "EC2_year": "split",
    "FlavorQCD": "split",
    "HF": "split",
    "HF_year": "split",
    "RelativeBal": "split",
    "RelativeSample_year": "split",
    
    #although pdf are added here
    "pdfscale": "pdfscale",
    
    #although JMS/JMR are added here
    "JMS"     : "JMS",
    "JMR"     : "JMR",
}


uncluste = {
    "UE": "unclusteredEnergy",
}


parser = argparse.ArgumentParser()
parser.add_argument(
    "--templates-dir",
    default="",
    type=str,
    help="input pickle file of dict of hist.Hist templates",
)

parser.add_argument(
    "--nTF1",
    default=None,
    nargs="*",
    type=int,
    help="order of polynomial for TF1.",
)
parser.add_argument(
    "--nTF2",
    default=None,
    nargs="*",
    type=int,
    help="order of polynomial for TF2.",
)
parser.add_argument(
    "--nTF3",
    default=None,
    nargs="*",
    type=int,
    help="order of polynomial for TF3.",
)

parser.add_argument("--cards-dir", default="cards", type=str, help="output card directory")
parser.add_argument("--model-name", default="HWWfhModel", type=str, help="output model name")
parser.add_argument("--mcstats-threshold", default=100, type=float, help="mcstats threshold n_eff")
parser.add_argument("--epsilon", default=1e-2, type=float, help="epsilon to avoid numerical errs")
parser.add_argument(
    "--scale-templates", default=None, type=float, help="scale all templates for bias tests"
)
parser.add_argument(
    "--min-qcd-val", default=1e-4, type=float, help="clip the pass QCD to above a minimum value"
)
parser.add_argument(
    "--year",
    help="year",
    type=str,
    default="all",
    choices=["2016APV", "2016", "2017", "2018", "all"],
)
add_bool_arg(parser, "mcstats", "add mc stats nuisances", default=True)
add_bool_arg(parser, "bblite", "use barlow-beeston-lite method", default=True)

args = parser.parse_args()

#create datacards for individual year
if args.year != "all":
    years = [args.year]
    full_lumi = LUMI[args.year]
else:
    full_lumi = np.sum(list(LUMI.values()))
    
if args.nTF1 is None:
    args.nTF1 = [0] # set default to 0

if args.nTF2 is None:
    args.nTF2 = [0] # set default to 0
     
if args.nTF3 is None:
    args.nTF3 = [0] # set default to 0

mc_samples = OrderedDict(
    [
        ("TT", "ttbar"),
        ("ST", "single_top"),
        ("WJets", "wjets"),
        ("Rest", "rest_bkg"),
    ]
)
bg_keys = list(mc_samples.keys())
sig_keys = [
    "Signal",
]
for key in sig_keys:
    mc_samples[key] = key
        
all_mc = list(mc_samples.keys())
logging.info("all MC = %s" % all_mc)


# dictionary of nuisance params -> (modifier, samples affected by it, value)
nuisance_params = {
    # LUMI: https://gitlab.cern.ch/hh/naming-conventions#experimental-uncertainties
    "lumi_13TeV_2016": Syst(
        prior="lnN", samples=all_mc, value=1.01 ** ((LUMI["2016"] + LUMI["2016APV"]) / full_lumi)
    ),
    "lumi_13TeV_2017": Syst(prior="lnN", samples=all_mc, value=1.02 ** (LUMI["2017"] / full_lumi)),
    "lumi_13TeV_2018": Syst(prior="lnN", samples=all_mc, value=1.015 ** (LUMI["2018"] / full_lumi)),
    "lumi_13TeV_correlated": Syst(
        prior="lnN",
        samples=all_mc,
        value=(
            (1.006 ** ((LUMI["2016"] + LUMI["2016APV"]) / full_lumi))
            * (1.009 ** (LUMI["2017"] / full_lumi))
            * (1.02 ** (LUMI["2018"] / full_lumi))
        ),
    ),
    "lumi_13TeV_1718": Syst(
        prior="lnN",
        samples=all_mc,
        value=((1.006 ** (LUMI["2017"] / full_lumi)) * (1.002 ** (LUMI["2018"] / full_lumi))),
    ),
    
    #assign 5% unc for Wjets, Top and rest background
    "WJets_rate": Syst(prior="lnN", samples=["WJets"], value=1.05),
    "TT_rate": Syst(prior="lnN", samples=["TT"], value=1.05),
    "ST_rate": Syst(prior="lnN", samples=["ST"], value=1.05),
    "Rest_rate": Syst(prior="lnN", samples=["Rest"], value=1.05),
    
}

if args.year != "all":
    # remove other years' keys
    for key in [
        "lumi_13TeV_2016",
        "lumi_13TeV_2017",
        "lumi_13TeV_2018",
        "lumi_13TeV_correlated",
        "lumi_13TeV_1718",
    ]:
        if key != ("lumi_13TeV_%s" % ((args.year))) and key not in ("lumi_13TeV_%s" % ((args.year))):
            del nuisance_params[key]

nuisance_params_dict = {
    param: rl.NuisanceParameter(param, syst.prior) for param, syst in nuisance_params.items()
}

# dictionary of correlated shape systematics: name in templates -> name in cards, etc.
corr_year_shape_systs = {
    # "triggerEffSF_uncorrelated": Syst(name="triggerEffSF_uncorrelated", prior="shape", samples=all_mc),
    "FSRPartonShower": Syst(name="ps_fsr", prior="shape", samples=all_mc),
    "ISRPartonShower": Syst(name="ps_isr", prior="shape", samples=all_mc),
    "QCDscale": Syst(
        name=("%s_QCDScale" % ((CMS_PARAMS_LABEL))),
        prior="shape",
        samples=bg_keys,
        samples_corr=False,
    ),
    "L1Prefiring": Syst(name="L1Prefiring", prior="shape", samples=all_mc),
    # "UE": Syst(name="unclustered_Energy", prior="shape", samples=all_mc),
    # "JES": Syst(name="CMS_scale_j", prior="shape", samples=all_mc),
    # "pdfscale": Syst(
    #     name="pdf", 
    #     prior="shape", 
    #     # samples=bg_keys,
    #     samples = bg_keys,
    #     samples_corr=False,
    #     ),
    #split JES
    "Absolute": Syst(name="CMS_scale_j_Abs", prior="shape", samples=all_mc),
    "BBEC1": Syst(name="CMS_scale_j_BBEC1", prior="shape", samples=all_mc),
    "EC2": Syst(name="CMS_scale_j_EC2", prior="shape", samples=all_mc),
    "FlavorQCD": Syst(name="CMS_scale_j_FlavQCD", prior="shape", samples=all_mc),
    "HF": Syst(name="CMS_scale_j_HF", prior="shape", samples=all_mc),
    "RelativeBal": Syst(name="CMS_scale_j_RelBal", prior="shape", samples=all_mc),
}

uncorr_year_shape_systs = {
    #split JES
    "Absolute_year": Syst(name="CMS_scale_j_Abs", prior="shape", samples=all_mc),
    "BBEC1_year": Syst(name="CMS_scale_j_BBEC1", prior="shape", samples=all_mc),
    "EC2": Syst(name="CMS_scale_j_EC2", prior="shape", samples=all_mc),
    "HF": Syst(name="CMS_scale_j_HF", prior="shape", samples=all_mc),
    "RelativeSample_year": Syst(name="CMS_scale_j_RelSample", prior="shape", samples=all_mc),
    
    "JMS": Syst(name=("%s_jms" % ((CMS_PARAMS_LABEL))), prior="shape", samples=all_mc),
    "JMR": Syst(name=("%s_jmr" % ((CMS_PARAMS_LABEL))), prior="shape", samples=all_mc),
    "JER": Syst(name="CMS_res_j", prior="shape", samples=all_mc),
    "pileup": Syst(name="CMS_pileup", prior="shape", samples=all_mc),
}

shape_systs_dict = {}
for skey, syst in corr_year_shape_systs.items():
    if not syst.samples_corr:
        # separate nuisance param for each affected sample
        for sample in syst.samples:
            if sample not in mc_samples:
                continue #means MC name error
            shape_systs_dict[("%s_%s" % ((skey), (sample)))] = rl.NuisanceParameter(
                ("%s_%s" % ((syst.name), (mc_samples[sample]))), "shape"
            )
    else:
        shape_systs_dict[skey] = rl.NuisanceParameter(syst.name, "shape")
    
for skey, syst in uncorr_year_shape_systs.items():
    for year in years:
        if year in syst.uncorr_years:
            shape_systs_dict[("%s_%s" % ((skey), (year)))] = rl.NuisanceParameter(
                ("%s_%s" % ((syst.name), (year))), "shape"
            )

def get_templates(
    templates_dir,
    years,
    scale = None,
):
    """Loads templates, combines bg and sig templates if separate, sums across all years"""
    templates_dict = {}

    for year in years:
        print("template_dir =",templates_dir )
        print("template =",templates_dir / ("templates_%s.pkl" % ((year))) )
        with (templates_dir / ("templates_%s.pkl" % ((year)))).open("rb") as f:
            templates_dict[year] = rem_neg(pkl.load(f))

    templates_summed = sum_templates(templates_dict, years)  # sum across years
    return templates_dict, templates_summed

def fill_regions(
    model,
    regions,
    templates_dict,
    templates_summed,
    mc_samples,
    nuisance_params,
    nuisance_params_dict,
    corr_year_shape_systs,
    uncorr_year_shape_systs,
    shape_systs_dict,
    bblite = True,
):
    """Fill samples per region including given rate, shape and mcstats systematics.
    Ties "blinded" and "nonblinded" mc stats parameters together.

    Args:
        model (rl.Model): rhalphalib model
        regions (List[str]): list of regions to fill
        templates_dict (Dict): dictionary of all templates
        templates_summed (Dict): dictionary of templates summed across years
        templates: the combination the above two options
        mc_samples (Dict[str, str]): dict of mc samples and their names in the given templates -> card names
        nuisance_params (Dict[str, Tuple]): dict of nuisance parameter names and tuple of their
          (modifier, samples affected by it, value)
        nuisance_params_dict (Dict[str, rl.NuisanceParameter]): dict of nuisance parameter names
          and NuisanceParameter object
        corr_year_shape_systs (Dict[str, Tuple]): dict of shape systs (correlated across years)
          and tuple of their (name in cards, samples affected by it)
        uncorr_year_shape_systs (Dict[str, Tuple]): dict of shape systs (unccorrelated across years)
          and tuple of their (name in cards, samples affected by it)
        shape_systs_dict (Dict[str, rl.NuisanceParameter]): dict of shape syst names and
          NuisanceParameter object
        pass_only (List[str]): list of systematics which are only applied in the pass region(s)
        bblite (bool): use Barlow-Beeston-lite method or not (single mcstats param across MC samples)
        mX_bin (int): if doing 2D fit (for resonant), which mX bin to be filled
    """

    for region in regions:
        region_templates = templates_summed[region]

        # pass region = SR1a, SR1b, SR2a, SR2b, SR3a, SR3b, and same with "Blinded" suffix
        pass_region = False
        pass_regs = ["SR1","SR2","SR3"]
        for pass_regi in pass_regs:
            if pass_regi in region: pass_region = True
        region_noblinded = region.split("Blinded")[0]
        logging.info("starting region: %s" % region)
        ch = rl.Channel(region.replace("_", "")) 
        print(region.replace("_", ""))
        model.addChannel(ch)
        blind_str = "Blinded" if region.endswith("Blinded") else ""

        for sample_name, card_name in mc_samples.items():
            # don't add signals in fail regions
            if sample_name in sig_keys:
                if not pass_region:
                    #don't need to enter CR signal anyway
                    logging.info(("\nSkipping %s in %s region\n" % ((sample_name), (region))))
                    continue

            logging.info("get templates for: %s" % sample_name)

            sample_template = region_templates[sample_name,:]

            stype = rl.Sample.SIGNAL if sample_name in sig_keys else rl.Sample.BACKGROUND
            sample = rl.TemplateSample(ch.name + "_" + card_name, stype, sample_template)

            # nominal values, errors
            values_nominal = np.maximum(sample_template.values(), 0.0) #select value >= 0

            mask = values_nominal > 0
            errors_nominal = np.ones_like(values_nominal)
            errors_nominal[mask] = (
                1.0 + np.sqrt(sample_template.variances()[mask]) / values_nominal[mask]
            )

            logging.debug("nominal   : {nominal}".format(nominal=values_nominal))
            logging.debug("error     : {errors}".format(errors=errors_nominal))

            #not used
            if not bblite and args.mcstats:
                # set mc stat uncs
                logging.info("setting autoMCStats for %s in %s" % (sample_name, region))
                # tie MC stats parameters together in blinded and "unblinded" region in nonresonant
                region_name = region_noblinded
                stats_sample_name = ("%s_%s_%s" % ((CMS_PARAMS_LABEL), (region_name), (card_name)))
                sample.autoMCStats(
                    sample_name=stats_sample_name,
                    # this function uses a different threshold convention from combine
                    threshold=np.sqrt(1 / args.mcstats_threshold),
                    epsilon=args.epsilon,
                )

            # rate systematics
            for skey, syst in nuisance_params.items():
                # pass
                if sample_name not in syst.samples or (not pass_region and syst.pass_only):
                    continue
                
                # for lp sf, should only be applied to corresponding regions, i.e., "a" or "b"
                if syst.pass_only and (syst.apply_reg not in region_noblinded):
                    continue 

                logging.info(("Getting %s rate" % ((skey))))

                param = nuisance_params_dict[skey] #rl.NuisanceParameter object

                val, val_down = syst.value, syst.value_down
                if syst.diff_regions:
                    region_name = region_noblinded 
                    val = val[region_name]
                    val_down = val_down[region_name] if val_down is not None else val_down
                if syst.diff_samples:
                    val = val[sample_name]
                    val_down = val_down[sample_name] if val_down is not None else val_down

                sample.setParamEffect(param, val, effect_down=val_down)

            # correlated shape systematics
            for skey, syst in corr_year_shape_systs.items():
                if sample_name not in syst.samples or (not pass_region and syst.pass_only):
                    continue

                logging.info(("Getting %s shapes" % ((skey))))

                if skey in jecs or skey in uncluste:
                    # JEC/UEs saved as different "region" in dict
                    up_hist = templates_summed[("%s_%s_up%s" % ((region_noblinded), (skey), (blind_str)))][sample_name,:]
                    down_hist = templates_summed[("%s_%s_down%s" % ((region_noblinded), (skey), (blind_str)))][sample_name,:]

                    values_up = up_hist.values()
                    values_down = down_hist.values()
                else:
                    # weight uncertainties saved as different "sample" in dict
                    values_up = region_templates[("%s_%s_up" % ((sample_name), (skey))), :].values()
                    values_down = region_templates[("%s_%s_down" % ((sample_name), (skey))), :].values()

                logger = logging.getLogger(("validate_shapes_%s_%s_%s" % ((region), (sample_name), (skey))))

                effect_up, effect_down = get_effect_updown(
                    values_nominal, values_up, values_down, mask, logger, args.epsilon
                )
                # logging.info(("final effect up is %s" % ((effect_up))))
                # logging.info(("final effect down is %s" % ((effect_down))))
                # separate syst if not correlated across samples
                sdkey = skey if syst.samples_corr else ("%s_%s" % ((skey), (sample_name)))
                sample.setParamEffect(shape_systs_dict[sdkey], effect_up, effect_down)


            # uncorrelated shape systematics
            for skey, syst in uncorr_year_shape_systs.items():
                if sample_name not in syst.samples or (not pass_region and syst.pass_only):
                    continue

                logging.info(("Getting %s shapes" % ((skey))))

                for year in years:
                    if year not in syst.uncorr_years:
                        continue
                    logging.info(("year: %s" % ((year))))
                    values_up, values_down = get_year_updown(
                        templates_dict,
                        sample_name,
                        region,
                        region_noblinded,
                        blind_str,
                        year,
                        skey,
                        years
                    )
                    #get summed templates with only the given year's shape shifted up and down by the ``skey`.
                    logger = logging.getLogger(("validate_shapes_%s_%s_%s_%s" % ((region), (sample_name), (skey), (year))))
                    
                    # logging.info(("final values up in %s is %s" % ((year), (values_up))))
                    # logging.info(("nom value in %s is %s" % ((year), (values_nominal))))
                    # logging.info(("final values down in %s is %s" % ((year), (values_down))))

                    effect_up, effect_down = get_effect_updown(
                        values_nominal, values_up, values_down, mask, logger, args.epsilon
                    )
                    # logging.info(("nom value in %s is %s" % ((year), (values_nominal))))
                    # logging.info(("final effect up in %s is %s" % ((year), (effect_up))))
                    # logging.info(("final effect down in %s is %s" % ((year), (effect_down))))
                    sample.setParamEffect(
                        shape_systs_dict[("%s_%s" % ((skey), (year)))], effect_up, effect_down
                    )

            ch.addSample(sample)

        # we always use bblite method
        if bblite and args.mcstats:
            # pass
            # tie MC stats parameters together in blinded and "unblinded" region in nonresonant
            channel_name = region_noblinded 
            ch.autoMCStats(
                channel_name=("%s_%s" % ((CMS_PARAMS_LABEL), (channel_name))),
                threshold=args.mcstats_threshold,
                epsilon=args.epsilon,
            )

        # data observed
        ch.setObservation(region_templates["data", :])

def alphabet_fit(
    model,
    shape_vars,
    templates_summed,
    scale = None,
    min_qcd_val = None,
):   
    # using SR1, SR2, SR3, CR below:
    shape_var = shape_vars[0]
    m_obs = rl.Observable(shape_var.name, shape_var.bins)
    
    qcd_eff_1 = (
        templates_summed[("SR1" % (()))]["QCD", :].sum().value
        / templates_summed[("CR" % (()))]["QCD", :].sum().value
    )
    qcd_eff_2 = (
        templates_summed[("SR2" % (()))]["QCD", :].sum().value
        / templates_summed[("CR" % (()))]["QCD", :].sum().value
    )
    qcd_eff_3 = (
        templates_summed[("SR3" % (()))]["QCD", :].sum().value
        / templates_summed[("CR" % (()))]["QCD", :].sum().value
    )

    
    # initialize transfer factor, value here won't influence final results
    # each SR should use one polynomial

    tf_dataResidual_1 = rl.BasisPoly(
        ("%s_tf_dataResidual_1" % ((CMS_PARAMS_LABEL))),
        (shape_var.order_1,),
        [shape_var.name],
        basis="Bernstein",
        limits=(-20, 20),
        square_params=True, 
    )
    tf_dataResidual_2 = rl.BasisPoly(
        ("%s_tf_dataResidual_2" % ((CMS_PARAMS_LABEL))),
        (shape_var.order_2,),
        [shape_var.name],
        basis="Bernstein",
        limits=(-20, 20),
        square_params=True, 
    )  
    tf_dataResidual_3 = rl.BasisPoly(
        ("%s_tf_dataResidual_2" % ((CMS_PARAMS_LABEL))),
        (shape_var.order_3,),
        [shape_var.name],
        basis="Bernstein",
        limits=(-20, 20),
        square_params=True, 
    )  
    
    # set TF parameters for each pass region(6 SRs)
    tf_dataResidual_params_1 = tf_dataResidual_1(shape_var.scaled)
    tf_dataResidual_params_2 = tf_dataResidual_2(shape_var.scaled)
    tf_dataResidual_params_3 = tf_dataResidual_3(shape_var.scaled)

    tf_params_pass_1 = qcd_eff_1 * tf_dataResidual_params_1  # scale params initially by qcd eff
    tf_params_pass_2 = qcd_eff_2 * tf_dataResidual_params_2  # scale params initially by qcd eff
    tf_params_pass_3 = qcd_eff_3 * tf_dataResidual_params_3  # scale params initially by qcd eff
            
    #set QCD parameters for 3 CRs
    qcd_params = np.array(
        [
            rl.IndependentParameter(("%s_tf_dataResidual_CR_Bin%s" % ((CMS_PARAMS_LABEL), (i))), 0)
            for i in range(m_obs.nbins)
        ]
    )
    
    for blind_str in ["", "Blinded"]:
        # for blind_str in ["Blinded"]:
        passChName1 = ("SR1%s" % ((blind_str))).replace("_", "")
        passChName2 = ("SR2%s" % ((blind_str))).replace("_", "")
        passChName3 = ("SR3%s" % ((blind_str))).replace("_", "")
        failChName = ("CR%s" % ((blind_str))).replace("_", "")
        
        #Get pass and fail channel information
        failCh = model[failChName]
        passCh1 = model[passChName1]
        passCh2 = model[passChName2]
        passCh3 = model[passChName3]
        
        # sideband fail
        # was integer, and numpy complained about subtracting float from it
        initial_qcd = failCh.getObservation().astype(float)
                
        for sample in failCh:
            if sample.sampletype == rl.Sample.SIGNAL:
                continue
            logging.debug("subtracting %s from qcd" % sample._name)
            initial_qcd -= sample.getExpectation(nominal=True)
            
        #avoid negative value after the subtraction
        if np.any(initial_qcd < 0.0):
            initial_qcd[np.where(initial_qcd < 0)] = 0
        
        # idea here is that the error should be 1/sqrt(N), so parametrizing it as (1 + 1/sqrt(N))^qcdparams
        # will result in qcdparams errors ~±1
        # but because qcd is poorly modelled we're scaling sigma scale

        sigmascale = 10  # to scale the deviation from initial, value >100 can make SR2a/SR2b/CR2 fit work
        if scale is not None:
            sigmascale *= scale

        scaled_params = (
            initial_qcd * (1 + sigmascale / np.maximum(1.0, np.sqrt(initial_qcd))) ** qcd_params
        )
        
        # add samples
        
        #Set fail region below
        fail_qcd = rl.ParametericSample(
            ("%s_%s_qcd_datadriven" % ((failChName), (CMS_PARAMS_LABEL))),
            rl.Sample.BACKGROUND,
            m_obs,
            scaled_params,
        )
        failCh.addSample(fail_qcd)
                
        #Set pass region below
        pass_qcd_1 = rl.TransferFactorSample(
            ("%s_%s_qcd_datadriven" % ((passChName1), (CMS_PARAMS_LABEL))),
            rl.Sample.BACKGROUND,
            tf_params_pass_1,
            fail_qcd,
            min_val=min_qcd_val,
        )
        
        passCh1.addSample(pass_qcd_1)      
          
        pass_qcd_2 = rl.TransferFactorSample(
            ("%s_%s_qcd_datadriven" % ((passChName2), (CMS_PARAMS_LABEL))),
            rl.Sample.BACKGROUND,
            tf_params_pass_2,
            fail_qcd,
            min_val=min_qcd_val,
        )        
        passCh2.addSample(pass_qcd_2)
        
        pass_qcd_3 = rl.TransferFactorSample(
            ("%s_%s_qcd_datadriven" % ((passChName3), (CMS_PARAMS_LABEL))),
            rl.Sample.BACKGROUND,
            tf_params_pass_3,
            fail_qcd,
            min_val=min_qcd_val,
        )
        
        passCh3.addSample(pass_qcd_3)      
          

def main(args):
    #print years
    print("Years are,",years)
    
    # all SRs and CRs
    regions = ["SR1","SR2","SR3","CR"]
    regions_blinded = [region + "Blinded" for region in regions]
    regions = regions +  regions_blinded #only use blinded results now
    cur_dir = os.getcwd()
    print("current dir = ",cur_dir)
    
    #normalized the path
    args.templates_dir = Path(args.templates_dir)
    args.cards_dir = Path(args.cards_dir)
    
    # templates per region per year, templates per region summed across years
    templates_dict, templates_summed = get_templates(
        args.templates_dir, years,  args.scale_templates
    )
        
    #apply lund plane sf
    model = rl.Model("Wcb")
    #random template from which to extract shape vars
    sample_templates = templates_summed[next(iter(templates_summed.keys()))]
    
    #MH_Reco for full-hadronic boosted HWW
    shape_vars = [
        ShapeVar(name=axis.name, bins=axis.edges, order_1=args.nTF1[i],order_2=args.nTF2[i],order_3=args.nTF3[i])
        for i, axis in enumerate(sample_templates.axes[1:]) #should be [1:] for boosted HWW analysis, because the 1st axes is mass
    ]
    # logging.info("shape_var = ",shape_vars)
    print("shape_var[0] info",shape_vars[0].name,shape_vars[0].scaled,shape_vars[0].bins)
    fill_args = [
        model,
        regions,
        templates_dict,
        templates_summed,
        mc_samples,
        nuisance_params,
        nuisance_params_dict,
        corr_year_shape_systs,
        uncorr_year_shape_systs,
        shape_systs_dict,
        args.bblite,
    ]
    fit_args = [model, shape_vars,templates_summed, args.scale_templates, args.min_qcd_val]
    print("now order_1 is",args.nTF1[0])
    print("now order_2 is",args.nTF2[0])
    print("now order_3 is",args.nTF3[0])
    fill_regions(*fill_args)
    alphabet_fit(*fit_args)
    
    logging.info("rendering combine model")

    os.system(("mkdir -p %s" % ((args.cards_dir))))

    out_dir = (
        os.path.join(str(args.cards_dir), args.model_name)
        if args.model_name is not None
        else args.cards_dir
    )
    model.renderCombine(out_dir)

    with open(("%s/model.pkl" % ((out_dir))), "wb") as fout:
        pkl.dump(model, fout, 2)  # use python 2 compatible protocol
        
main(args)
