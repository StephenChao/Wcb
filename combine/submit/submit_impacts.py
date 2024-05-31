#!/usr/bin/python

"""
Splits toy generation into separate condor jobs and fits lowest order + 1 models for F-tests.

Author(s): Raghav Kansal, Yuzhe Zhao
"""
from __future__ import print_function

import argparse
import os
from pathlib import Path

import ROOT
from utils import parse_common_args, setup

import run_utils
from run_utils import add_bool_arg


def _tolist(argset):
    return [x.GetName() for x in argset]


def getParameters():
    """Get nuisance parameters from workspace"""
    f = ROOT.TFile.Open("combined_withmasks.root", "READ")
    w = f.Get("w")
    ps = _tolist(w.allVars())
    pois = _tolist(w.set("ModelConfig_POI"))
    obs = _tolist(w.genobj("ModelConfig").GetObservables())

    ret_ps = []
    for p in ps:
        if not (
            # "mcstat" in p
            # "dataResidual_CR" in p
            "qcdparam" in p
            or p.endswith(("_In", "__norm"))
            or p.startswith(("n_exp_", "mask_"))
            or p in pois
            or p in obs
        ):
            ret_ps.append(p)

    return ret_ps


def main(args):
    t2_local_prefix, t2_prefix, proxy, username, submitdir = setup(args)

    prefix = "impacts"
    local_dir = Path(("condor/impacts/%s/" % ((args.tag))))

    # make local directory
    logdir = local_dir / "logs"
    logdir.mkdir(parents=True, exist_ok=True)

    jdl_templ = ("%s/submit_impacts.templ.jdl" % ((submitdir)))
    sh_templ = ("%s/submit_impacts.templ.sh" % ((submitdir)))

    ps = getParameters()
    print(("Running impacts on %s parameters:" % ((len(ps)))))
    print(*ps, sep="\n")

    res_str = "-r" if args.resonant else ""
    
    #when running combination
    commands = [("run_blinded.sh %s -i" % ((res_str)))] + [
        ("run_blinded.sh %s --impactsf %s" % ((res_str), (p))) for p in ps
    ]

    #when running individually
    # commands = [("run_blinded.sh %s -i" % ((res_str)))] + [
    #     ("run_blinded.sh %s --impactsf %s" % ((res_str), (p))) for p in ps
    # ]
    
    impactfiles = ["higgsCombine_initialFit_impacts.MultiDimFit.mH125.root"] + [
        ("higgsCombine_paramFit_impacts_%s.MultiDimFit.mH125.root" % ((p))) for p in ps
    ]

    collect_command = ("/ospool/cms-user/yuzhe/Wcb/Wcb/combine/scripts/run_blinded.sh %s --impactsc %s" % ((res_str), (','.join(ps))))
    # collect_command = ("/ospool/cms-user/yuzhe/NanoNtupleChain/boostedHWW/combine/combination/scripts/run_combined.sh %s --impactsc %s" % ((res_str), (','.join(ps))))
    # collect_command = ("run_blinded.sh %s --impactsc %s" % ((res_str), (','.join(ps))))

    if args.local:
        print("Running locally")
        for command in commands:
            os.system(command)
        os.system(collect_command)
        return

    collect_sh = Path(("%s/collect.sh" % ((local_dir))))
    with collect_sh.open("w") as f:
        f.write(collect_command)

    os.system(("chmod u+x %s" % ((collect_sh))))
    print(("To collect impacts afterwards, run: %s" % ((collect_sh))))

    # submit jobs
    if args.submit:
        print("Submitting jobs")

    for p, command, impactfile in zip(["init"] + ps, commands, impactfiles):
        local_jdl = Path(("%s/%s_%s.jdl" % ((local_dir), (prefix), (p))))
        local_log = Path(("%s/%s_%s.log" % ((local_dir), (prefix), (p))))
        jdl_args = {
            "dir": local_dir,
            "prefix": prefix,
            "jobid": p,
            "proxy": proxy,
            "impactfile": impactfile,
        }
        run_utils.write_template(jdl_templ, local_jdl, jdl_args)

        localsh = ("%s/%s_%s.sh" % ((local_dir), (prefix), (p)))
        sh_args = {"command": "./" + command}
        run_utils.write_template(sh_templ, localsh, sh_args)
        os.system(("chmod u+x %s" % ((localsh))))

        if local_log.exists():
            local_log.unlink()

        if args.submit:
            os.system(("condor_submit %s" % ((local_jdl))))
        else:
            print("To submit ", local_jdl)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parse_common_args(parser)
    add_bool_arg(parser, "local", help="run locally", default=False)
    args = parser.parse_args()
    main(args)
