universe=Vanilla
+ProjectName = "cms.org"
requirements = (OpSysAndVer =?= "CentOS7")
RequestMemory = 2048
RequestCpus = 1
executable=/home/legao/Wcb/condor/production/NanoNtupleChain_25_Apr_2024/JetHT_Run2018A-UL2018_MiniAODv2-v1/exe_CUSTNANO_UL18_MINIAODv2_DATA.sh
transfer_executable=True
transfer_input_files=/home/legao/Wcb/condor/x509up_u14433,/home/legao/Wcb/condor/Scripts/fetchFiles.py
transfer_output_files=""
log=/home/legao/Wcb/condor/production/NanoNtupleChain_25_Apr_2024/JetHT_Run2018A-UL2018_MiniAODv2-v1/log//$(Cluster).log
output=/home/legao/Wcb/condor/production/NanoNtupleChain_25_Apr_2024/JetHT_Run2018A-UL2018_MiniAODv2-v1/log//$(Cluster).out
error=/home/legao/Wcb/condor/production/NanoNtupleChain_25_Apr_2024/JetHT_Run2018A-UL2018_MiniAODv2-v1/log//$(Cluster).err
notification=Never
should_transfer_files = YES
when_to_transfer_output = ON_EXIT
x509userproxy=x509up_u14433
+MaxRuntime=150000
on_exit_remove        = (ExitBySignal == False) && (ExitCode == 0)
on_exit_hold          = (ExitBySignal == True) || (ExitCode != 0)
on_exit_hold_reason   = strcat("Job held by ON_EXIT_HOLD due to ", ifThenElse((ExitBySignal == True), strcat("exit signal ", ExitSignal), strcat("exit code ", ExitCode)), ".")
periodic_release      = (NumJobStarts < 3) && ((CurrentTime - EnteredCurrentStatus) > 60*60)
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260001/11E388F0-821C-6B44-9B98-85935C790C96.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_593.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260000/557E51DC-DFA2-D749-8FAB-4A8894708C35.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_643.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260001/52CCBEF0-4658-7640-9FB1-728554B31F45.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_592.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/280000/B57F196C-3775-8942-AD9B-91FA0252F08F.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_344.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/280000/D5B283F8-147D-7B49-A170-BDF5BFD0265C.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_345.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/280000/49BAF21A-A7D4-6E4E-8232-0774FF7D0926.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_346.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/280000/1CD98F1C-C7BF-844F-9688-CDBAC5B31321.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_347.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/280000/F720B976-D740-424C-92E3-522B198DA101.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_340.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/280000/26CBB5BC-663F-714C-8476-549CF6A94DC2.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_341.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/280000/EEC457DE-4B74-E344-8DFA-F282E72B47D1.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_342.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/280000/E53D1CC6-4CB9-FF4F-A9B4-50D02AF9A71A.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_343.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/280000/489A54B4-C92F-444C-93DE-2A65164DF44F.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_348.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/280000/58015DE9-F383-2C4B-AE88-D0716CABDD14.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_349.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260000/020C4D59-9FF4-BA4C-B24C-6B77F9F0CC20.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_1296.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260001/C14AC18A-1364-AC43-80B6-038F02BC4808.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_594.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/50000/BE74F71C-E96A-864B-8006-BE5A6B5DA07C.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_2318.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/50000/94E79508-3C87-BF4E-9DEE-9A4C2927819E.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_2319.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/50000/DBA66A90-7164-7B4F-B041-63143C95840C.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_2316.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/50000/CA1B651F-CB02-CD42-B22B-61CB12C67E18.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_2317.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/50000/71FB91F6-4507-7449-983A-7FB881246097.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_2314.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/50000/0052533A-1B08-6744-AFA4-5F5F205184DC.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_2315.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/50000/B8A3454A-09D7-AC4B-898C-F119860A7111.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_2312.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/50000/39B4D313-A20F-0545-AA3A-A4791392C0AC.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_2313.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/50000/02853103-0B92-2540-A515-59D5C46C7CF4.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_2310.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/50000/D6F3F055-AF8A-B14C-A62F-6F1E31E09116.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_2311.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/280000/03EB0586-AD31-474C-AE83-559D13F9BB21.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_298.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/280000/E08BA7BC-EF05-E749-9A22-86F373DEE75C.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_299.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/280000/2CE33C89-3473-3C42-BC92-B66E9F348DBE.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_296.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/280000/89EBA10D-8969-F648-B16E-BAE90D7BEC46.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_297.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/280000/B133FBFB-942F-BB4E-BD0B-6C68BBDC500D.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_294.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/280000/99BEB8C1-BB91-374C-BBBF-58B16951249F.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_295.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/280000/0DA5C260-4283-1445-8251-9B5A3D2466A8.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_292.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/280000/86C72AC1-0902-8844-8B66-539DC398BDFA.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_293.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/280000/AE1B75E9-4882-B841-95A8-BDB6E87F8FD4.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_290.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/280000/548DFA03-4C18-FC42-9D8E-091C78BB901B.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_291.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/280001/7635AF59-645C-A347-8C74-725F053DC496.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_270.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/280001/508E30E0-F586-4447-8D89-03E1E6FB03E1.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_271.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/280001/AF2815E0-291B-EF45-8C53-D54D327F7C21.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_272.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/280001/902500DE-E04D-BC42-82FB-C69650A442F1.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_273.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/280001/12F1E720-7F71-CC48-AED5-2128768E61A8.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_274.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/280001/E3E8164A-8F7E-C741-A7B9-D98AB3413D80.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_275.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/280001/994595D4-D55E-FE4C-AB62-0236C1D6D0BF.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_276.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/280001/DA6B0095-B488-B14C-BB6C-2EA521816CC2.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_277.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/280001/75E9499B-20CC-8440-8D8E-9E2871A8A0B0.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_278.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/280001/B208D6A5-DA09-8E4C-82D0-7054C07EC2A9.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_279.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260001/8A7C9442-1269-7946-ABB2-2EDC64BA9EFB.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_738.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/270000/5687EDEA-B882-C345-9568-0CA3BA64CA4F.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_2268.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/270000/10AEEF32-0B96-3C4E-8BFC-E8F4B2C7E986.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_2269.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/270000/94A95818-02A4-3F4E-94A6-E8D52B88C726.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_2262.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/270000/FC4337A8-07B0-4F41-BE07-A94C4297CEAE.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_2263.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/270000/3B7BBDDF-D6F4-1B45-8D45-C56319934883.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_2260.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/270000/9183C371-7BD9-C34D-8B46-8DA561030012.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_2261.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/270000/FF038F81-655F-7B40-9D21-8B6FD6DA70D0.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_2266.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/270000/3346219E-1AD0-5A4A-A9CC-9FD9C58DA902.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_2267.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/270000/A266BFC3-65E0-5543-A9DE-55F47E3A87BA.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_2264.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/270000/29873AEB-042B-114E-8FF8-5EE344794A45.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_2265.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/280001/7F91D6BF-4DD1-E740-A0A7-7E81A7061AF1.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_2442.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/280001/6B8841CF-8ACB-0542-9753-1C364D5444BB.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_2443.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/280001/EE468549-62FB-AA46-B129-85CB776136CF.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_2440.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/280001/0EEEF255-3187-A042-99EB-7E97435A3C47.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_2441.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/280001/71B3C7C9-DED2-C64C-9C64-9F9BEECA545E.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_2446.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/280001/8277153B-4D5E-B54F-A542-B85B7B8924D1.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_2447.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/280001/35D4E27A-F74E-A545-A67E-F50F4A68E8F2.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_2444.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/280001/FE18EE23-01B5-0E4E-8587-2AB88EADC0E6.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_2445.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/280001/3C1BC1A2-A497-5F4C-B766-EEAAEC0153C8.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_2448.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/280001/136A432A-A463-164C-B355-370775146279.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_2449.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/270000/CC707EBD-5CA4-DC47-B4C0-6FDF28D782F0.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_2189.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/280000/1B865F1B-C375-1C4F-B651-A37C0B632D25.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_108.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/280000/31D9CE86-58CA-CD4F-86C0-BA68D3165974.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_109.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/280000/20AB959B-173F-FD45-8A57-FE591179EB37.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_102.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/280000/06D5D64E-6FD5-AD4F-9905-0782A64B984A.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_103.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/280000/54DF51AE-6610-0E41-B198-40426A1C9078.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_100.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/280000/5F32F6AA-036C-E948-8EF4-56C2307188D2.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_101.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/280000/3E26B943-7EF3-AB42-92A4-0F4FB724EB3A.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_106.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/280000/04A2FF22-2202-5B46-98B9-13B5DEC7E81E.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_107.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/280000/060832A2-CE95-9447-843A-B54FCCD89265.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_104.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/280000/5E03029B-80AB-0F49-A721-EFEACEEF5133.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_105.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/50000/1BD0F7BD-A4E7-5B4F-A1B6-080F5D0BA62C.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_2046.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/50000/CC53B0F5-28AD-F744-B8B7-F623888CA2A1.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_2047.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/50000/48FB73C9-7894-3048-BC38-8132521AA1D4.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_2044.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/50000/6A2FF58B-187D-F94A-88F6-47804389AC9E.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_2045.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/50000/082A8CD2-5B45-E04D-B8A5-BB4465630CBC.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_2042.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/50000/C6D76E6A-A065-8343-9E2D-9671C48965D3.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_2043.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/50000/B2D5F538-D3CC-2047-AF95-A02C7086BE8C.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_2040.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/50000/5E214125-6C8B-4944-8DD8-7DBF04C5A563.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_2041.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260000/5D4BC0E2-4B44-A647-83BB-AD4AC648A903.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_1297.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/50000/A4E852C1-1E73-F34E-A35D-32A4D2E3A141.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_2048.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/50000/0B31F48E-79B6-8C47-B0DC-A04D4D54612C.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_2049.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260000/F96640FA-DC21-B044-9881-F1471A697844.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_1210.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/40000/59335964-839B-7140-824A-3ECD3DF0A30D.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_1375.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/40000/EDF760EF-FD70-954C-BC1A-54CE124930EE.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_2688.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/40000/BC79926C-633B-8D4B-94ED-26F07E45AEFA.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_2689.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/40000/33B281FE-EB2F-8E40-B0F4-5016FC22D6A1.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_2684.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/40000/97E2164E-89A5-054D-BF77-A10A3A099B98.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_2685.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/40000/5687F422-EE05-0549-A0BE-A865A793C0D2.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_2686.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/40000/06B2866C-C91D-2549-A896-BC4BE3F51DBD.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_2687.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/40000/AA04E72A-3EE9-8F4C-927B-06BFB9A22C5D.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_2680.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/40000/7F098E61-5A7C-3D4A-8446-54F4926D1B35.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_2681.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/40000/01BF424C-16A6-5848-A23C-925529F217C7.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_2682.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/40000/AEC06E1F-F357-1A46-BBE2-56EC7EB6DEDA.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_2683.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/40000/9510CAB9-2545-3149-A1D9-295B5613A15E.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_1374.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/280000/D0EE89E9-A771-BB4F-A178-BE10B8654503.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_99.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/280000/83E54B33-F0C9-B041-93AC-FAA6B7068601.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_98.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/280000/8CD39165-F38F-254C-AD67-567B0FA7B733.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_91.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/280000/69DF5E14-A61A-9543-93BB-C7785523BCF8.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_90.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/280000/2A108942-5986-9344-A778-907F047A6E3A.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_93.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/280000/611D0D61-0CBE-2F49-97D1-E56199C1458B.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_92.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/280000/C372D7A9-7098-1049-ACC0-D78ECD873D26.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_95.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/280000/5624765D-BA5C-9843-9DC8-A1BC6DECFE01.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_94.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/280000/FFFE7DE5-BB5A-CE4D-8575-EDCF390326F6.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_97.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/280000/2954798E-D610-9444-89E5-52A59315DA8F.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_96.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260000/CC7F76B9-7A8C-D342-A990-2D946D41E172.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_1623.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/50000/04C26600-FF0A-5645-9C81-A2888498E84B.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_1990.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/50000/7C97655B-D1CD-D849-A12E-887CB58E13AC.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_1993.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260000/FE4B4FA5-91AD-E243-978B-3C6C24D7E5F5.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_1620.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260000/4A25ABB2-DC64-054D-8904-D17CD08E2F61.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_1627.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/50000/7E262C13-92A1-CF4C-8A46-A214EA7180C6.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_1994.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260000/ECD3AEFC-717E-024E-A99F-43980D52B7A7.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_1625.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260000/AAB56622-BB2E-0744-B020-92DB5586F4E6.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_1624.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/50000/259115C9-0AFA-DB4A-9003-56F76105B60D.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_1999.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/50000/AC9CC626-2CE6-024A-9F76-F94DAE512146.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_1998.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260000/FCE4F8F0-C398-7F4A-BCF4-81356D5D9661.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_1629.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260000/9F6A9D07-30D1-E04C-B86A-81C23C850165.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_1628.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260000/B620800A-8C97-EE43-9135-81EF3EF998EF.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_1293.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260001/04E6E982-0E12-C742-BE00-7E6ECF9C5CA8.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_559.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260001/D34E6AAF-28E0-4A44-9EED-DCE55F10C098.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_558.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260001/25387954-0DE1-FD45-B98D-476A645E6008.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_555.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260001/6729DC86-33E6-C24D-BD9D-F5048212514C.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_554.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260001/9E3003F6-72C4-3741-9588-7071B659498F.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_557.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260001/DE5D2CA6-E0F2-A943-847D-B2A9BA4142F0.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_556.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260001/FD1A5975-6A71-704B-9497-8A7785D7A75F.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_551.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260000/4A3D97C2-6EBB-C14E-AC4B-8CD474B6817D.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_550.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260001/064E37E0-E662-694B-9567-AF8C6B72F59E.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_553.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260001/6DDBBA27-E1A5-1F43-92DF-52E72048B187.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_552.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/270000/2AC03708-E096-4E42-B3DC-EE97613233F2.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_1199.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/270000/5227E6FD-AB6C-C342-8E87-9364EC0EAB9E.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_1198.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/40000/6A4685BE-E341-C541-AAC7-4E5854435737.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_1365.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/270000/BD7F3559-9665-7B4D-9AB9-8C6A69606B53.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_1191.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/270000/6B41FEB4-0F68-8141-98D6-2FE8199B518B.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_1190.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/270000/D3F7858A-AD95-4741-B43D-4DA27CF1D2D4.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_1193.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/270000/469651DF-70D2-CE4D-85D5-747F12ECD5F2.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_1192.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/270000/881E6AAA-8B8C-AE4D-8F84-2B392CF10211.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_1195.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/270000/8CC7B476-AE8F-5144-985C-AD05D642028B.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_1194.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/270000/AFD969C5-CFEF-474F-8C5A-4657F6AC87FE.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_1197.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/270000/AD7C102D-FE60-F441-87B4-453969819EB8.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_1196.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/50000/F0DCEA4D-FE04-F742-9457-20FBC5C14A64.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_1759.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/50000/FDD8956C-D19F-B743-BF93-1FC69B67CA70.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_1758.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/50000/1CD4728C-747C-C74D-B120-5548AB5EE891.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_1757.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/50000/17FF389B-072E-B44C-BEB5-18DD622E7FD4.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_1756.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/270000/907EA0A6-9E36-2644-B629-E900AC80CFEE.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_1755.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/270000/D58EE269-AC06-8349-ACC5-91CE761DC000.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_1754.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/50000/4D830F61-FB04-F744-BFE5-27C919B2E785.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_1753.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/50000/74FB5D61-7BEC-2641-817F-62D9626704D1.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_1752.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/50000/50350BF9-8088-4B4A-9CE0-81650F78E02C.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_1751.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/50000/3072BD47-1C2D-B244-B346-B56BA3825043.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_1750.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/270000/3CEF0C72-C421-3440-86D4-25A91F74A6FC.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_1177.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/270000/9CFF30A6-3FA7-6C46-86F2-A1279F3CE172.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_1176.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260000/851A8FFC-9374-2C48-8FDB-ACBC0050A2EC.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_1175.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260000/56DB4D7D-3F51-BB45-AE16-BA265551226C.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_1174.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260000/27DF274B-6FB2-634E-8713-B5B2021D2E34.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_1173.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260000/9342AB86-86A2-E940-8286-4348D732BB5B.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_1172.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260000/ABE5868A-ECDD-7246-882A-8C7837D185B5.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_1171.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260000/A3462068-7CC5-D24A-817D-0115C544A318.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_1170.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/270000/FF0269EA-95BF-3C47-B7BB-602808F3B3B8.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_1179.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/270000/94814220-8693-574C-B638-FE7DB220F45A.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_1178.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260001/178AB2D2-7A4A-0F4C-9573-A205FF8BC335.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_511.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260001/1D0B2481-DC40-D540-A002-6470DCCCC698.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_510.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260001/D087D3A4-357D-0C40-A7D8-18FC88C4AF2C.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_513.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260001/738905CB-1F7E-0D43-A289-8423FCD34055.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_644.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/40000/C606EC98-9F14-534E-A04F-5533656E43D8.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_1285.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/40000/6AC5CCDA-4FF2-8945-81FB-0A0D968B41E3.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_1284.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/40000/B048AFC5-2C58-FE4B-9FDD-6A43A76EC158.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_1287.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/40000/461D2D66-1EF6-0342-89B0-7498522A07E6.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_1286.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/40000/0A02D397-B94A-7249-B2E8-813512F77F9D.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_1281.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/40000/1619215F-8D89-5547-AE64-8AAD9A19F7A2.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_1280.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/40000/F5D2ED45-639E-DF41-B0AC-ED4C5386B6BC.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_1283.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/40000/F0B2C4B5-8BD6-334C-885B-62858A0B3B2C.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_1282.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260001/1550D6CA-AF6D-6E47-BB66-63C2FF92A13B.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_515.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/40000/705A183C-D05D-2746-9146-95A7F69402A7.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_1289.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/40000/E8EFE851-05A5-3C44-9A7E-7AD31BC1E9EF.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_1288.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260001/9DE2F904-F98D-E049-AC8F-7ED801C962FB.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_514.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/40000/6A6CCC98-75B6-D44D-BD5C-D1EAA1771070.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_1579.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/40000/D0E239E3-F144-EE43-833E-DCE58974474B.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_1578.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260001/88AFF270-87AC-8D45-B79A-C5FAA850BC2D.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_689.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260001/00C2361E-476F-A943-921B-81001DDD03EC.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_688.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260001/07BEF709-9B24-E949-83D4-5245DA6AC6C2.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_685.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260001/19122C61-0380-8447-B56E-F349F204F737.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_684.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260001/38FC2F0E-6AB0-6E46-B153-AEDCAC1F7841.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_687.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260001/063C8B7B-C5A1-7C4E-9707-C21D6A2ED126.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_686.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260001/78B6CBF9-6F2B-F742-B564-A89C66B3C73D.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_681.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260001/F7D60E23-F0B9-A741-BBDE-62EA60F8CD6F.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_680.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260001/36ED69CB-EABA-084F-B048-35FA08CBADDB.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_683.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260001/7B2AEE7D-1BA3-B742-BA22-C199CCCA4F39.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_682.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260001/15016A35-299A-7749-868C-9859195BE94E.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_623.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260000/3DF4662C-C9CB-0645-A8A0-C7B1F412E9F5.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_819.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260001/9596D5E1-4C44-604F-BC27-AC49847BD7E7.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_622.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260000/51A5CE7C-FB38-1546-980D-07843FEFB568.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_1240.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260001/CFD04398-72FD-B14B-BF3A-53AE58765B1D.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_621.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260000/A0AC0DE8-7F96-D843-B37A-DC8EAAB1E786.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_1224.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260001/BB82390D-EE48-D64A-8F4E-387D204C498A.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_627.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260001/E5277CE7-336C-D247-BC0E-DEB272F14AE7.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_626.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260000/D163FA23-FD36-B547-9E19-D8F5C279C9E8.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_1221.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260000/8758E944-B214-0C4F-9693-F4D0AEB42F39.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_1243.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260001/3E93F9D1-4A06-1D46-9E18-AEE0EAC90764.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_624.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/280000/17C9F474-EC20-CE48-8D2D-9C23380D8463.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_407.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/40000/09AA0383-4E29-AB47-A3A7-4293CE90B983.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_1370.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/280000/18D3FA10-8CD1-2C46-A06C-9BA8CC1677FB.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_405.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/280000/9D0D4FBC-64FD-554D-969F-3553FDB12552.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_404.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/280000/B1E6DEFB-5064-D442-93BE-D23385E1B186.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_403.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/280000/3BF916D1-809E-9748-B49C-177AEBBF2900.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_402.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/40000/2E937C0A-421A-9042-AA72-F4348A23AC11.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_1377.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/280000/C2AD3338-B432-D242-9B8E-309B4A0B34A7.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_400.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/40000/36CE84C3-6C03-024B-B5DA-3A0019F2CC0F.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_1379.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/40000/0914CF93-0286-FB4D-9E78-2E559B27CAAB.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_1378.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/280000/2EC50538-5026-304D-A229-B1528F7E7214.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_452.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/280000/524EE2DA-E9A7-814D-9C4B-9399B2354FBF.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_409.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/280000/A7CEB5A9-951A-374F-A2BB-60A58FC4D82E.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_408.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/270000/B3054F0C-AC52-524A-A1BB-442D2D0E5CCD.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_1343.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/270000/E7D557CB-EB14-4F4A-B2E4-2F420A1738FA.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_1344.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/270000/10631C66-F2CC-5F47-95B4-86D274F9F7AB.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_2192.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/270000/79A18072-4743-6F40-9655-1FDEDD214E89.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_1345.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/280000/8A2ACBF6-A790-4146-8D97-27C47FC7CA8C.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_456.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/280000/6756D902-8A7F-8C4F-8873-35ACCBF4244B.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_457.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260000/7AE2391C-EBF9-554D-9E40-2667999257E6.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_1018.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/280000/9978D374-91B7-E847-8B72-CE0B866D599C.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_379.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/280000/7FFF72DE-26FF-C440-91BC-27F488855A3E.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_378.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260001/88E50731-3F58-DA45-A66B-654446298EDE.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_647.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/280000/95078821-D1EF-0443-AC99-515928ACB90F.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_371.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/280000/6B04A0F4-2471-5F47-92F9-211F645109F9.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_370.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/280000/2417D2E5-DF34-814F-89B5-40C25CD2E8AD.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_373.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/280000/88CD16FC-0036-D94D-ACFD-8BD487D0D887.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_372.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/280000/07D90E4C-0605-274B-A06D-D804BB2A6944.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_375.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/280000/2560C0A2-C1AE-754E-A3E9-38C03EC0B65F.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_374.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/280000/999A440F-37F9-524F-A9FB-7F4BBCC0C57F.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_377.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/280000/B00B9145-5C35-084B-B850-CEC74B1BC169.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_376.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/270000/2B4E5984-C2BE-DB4D-B07E-93EF3E60B1F5.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_1244.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/280000/58E557BF-40CB-4E43-8A33-5AECF7ADBA98.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_393.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/280000/6BB95247-E1D2-4446-8035-9B5599B27290.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_392.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/280000/947F33EA-F95A-504E-A846-65B056C63562.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_391.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/280000/77BE4EC2-B691-A749-BFA5-D603405E745E.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_390.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/280000/37153A01-CFBC-F646-A1B9-2E1C86919BD2.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_397.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/280000/03855B95-9432-FA48-8A2A-59E642568630.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_396.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/280000/DD4B92D3-B4EE-974C-A91F-D0212353BB42.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_395.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/280000/54D0A888-908F-B74F-A4DD-E452A32C4634.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_394.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/280000/EB2399F7-3ADD-6D4C-8181-CDD4751C83E4.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_399.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/280000/3D7F069F-DDC0-C646-8318-EC0333302B51.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_398.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/50000/128123A2-A23E-134E-B281-FD5767267636.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_2309.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/50000/C46BED11-26C7-A448-A02E-B7A1C3AAAC11.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_2308.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/50000/2AC607BA-3D1F-B54D-8C94-A24ABC2FC111.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_2301.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/50000/C01CA775-2810-9F4F-BA60-F305F9EC90B5.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_2300.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/50000/3585FD09-AD88-9A45-911B-C9F9CAFC75BE.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_2303.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/50000/8186DF83-C872-054E-B724-E5CB3A565D49.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_2302.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/50000/F2755142-D304-2944-A9F8-0BE2E3855C83.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_2305.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/50000/31A4468E-6442-B547-8215-B96717376A95.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_2304.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/50000/2FF01BA5-18D6-544C-B550-9F8A06442CFF.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_2307.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/50000/EB5D1C65-D907-4443-9607-595514BEF683.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_2306.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/270000/E5410AA6-5B22-7C41-9C65-88962D72C85A.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_1246.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/280000/5B0752B7-7B0F-184C-9BEB-50FE68B65ACA.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_245.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/280000/D0342799-4EEA-E943-87C7-BDC6C7A8A7B5.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_244.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/280001/E961CB36-0320-CF48-B5C9-58B85DD99298.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_247.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/280000/8DC03EC5-D613-FB42-9686-29F481949238.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_246.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/280000/04738059-B88D-3D4D-B908-4DE7CE25805C.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_241.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/280000/3399F906-F75A-0841-8C7F-4BFD8691FCAD.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_240.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/280000/057CEF32-B3A7-2544-B90C-66285AE0E28F.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_243.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/280000/6498B4D3-177F-004F-BE16-8F9D6C10B592.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_242.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/280001/6F2EC9CD-4A6A-2C41-9D9E-5ADD3BA74AE3.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_249.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/280001/AD2E3617-F2FD-A946-ACDF-3D43578D4450.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_248.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/270000/AAB70C36-C8AA-804C-8097-A829AB759C77.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_2275.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/270000/10FD9EAE-D00A-A846-8EC6-56B4DFCFC00B.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_2274.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/270000/D3A6D4DE-C9E9-824D-AB3A-490D15D1EEFD.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_2277.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/270000/CC9EFB4A-8633-E04F-BC69-953595F390FD.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_2276.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/270000/D1731958-8BFC-C248-87B2-07413CD33573.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_2271.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/270000/26016E96-EEBC-C447-88E9-67949D9C7831.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_2270.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/270000/47D68ACF-D66F-584F-B342-00CB23217506.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_2273.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/270000/7A969B7F-BBD9-D044-BDB6-3C05AE84A2D2.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_2272.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/270000/9FD5B1B3-5D4B-9B42-A891-BCD69E089513.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_2279.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/270000/D4B65D2B-CA62-0F42-BA23-656E1BF9819E.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_2278.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/280001/200D6664-7D10-2742-812F-5C6242E3366C.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_2459.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/280001/26D0D062-1A6B-264A-BE61-720829409794.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_2458.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/270000/554AD2B3-7F8E-C441-B39C-CD037619C7C2.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_2156.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/280001/914C08DC-CF80-0E4C-AEC2-B38A7B2BD1F1.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_2455.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/280001/0552C976-9EFF-AB43-A523-27ACC6F64A67.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_2454.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/280001/9EBDBE72-6229-7E4B-8124-61A730655F44.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_2457.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/280001/20C34128-E374-804A-81EE-7B180DCBAA86.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_2456.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/280001/F4032321-314F-C74C-9B12-63E59F730C94.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_2451.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/280001/C7AF3E27-38C7-1E43-8688-DFDFFAC06E34.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_2450.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/280001/81A3043E-5C74-D443-9196-19E2521CC9CB.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_2453.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/280001/0C7E131F-F26B-E44F-926A-B1F08A710AF4.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_2452.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/280000/A8378AF9-90B1-7D43-9E07-5ECD8DB6F661.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_179.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/280000/0EF416A7-E47D-1744-B3D4-1026E30C67BD.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_178.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/280000/7CF165B5-AC32-7947-9182-0D921FECE84F.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_177.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/280000/6C3295FB-2094-E04E-9D4D-3AE40EEDCBB0.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_176.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/280000/934FAE42-16CC-CD45-BDEF-B0AE83EC4AF9.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_175.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/280000/DD935F74-3A32-8949-AB59-BC5532721F57.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_174.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/280000/D840333D-C887-3945-B5B7-D5867D58BE3A.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_173.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/280000/0F7CF3FD-B8F3-B842-874C-13EF2E8E404D.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_172.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/280000/8AE12D17-1077-F346-89A4-57636787E5BC.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_171.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/280000/1E044B8A-F15E-A842-B86D-585DF64C4623.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_170.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/50000/C43F6CEA-5934-4149-99B5-8BCADC3AF8BB.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_2051.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/50000/3323FCD6-6E8F-F849-A918-61E22685CFD3.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_2050.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/50000/6481598F-91F5-DE41-AAB2-6D3A377AF8B9.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_2053.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/50000/BE30BA11-80D0-AC40-AD82-716C9932687F.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_2052.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/50000/3506CCAF-7493-E248-801E-1C442CB6FA21.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_2055.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/50000/8AD3ADF9-E2D0-1C44-A4D2-D47AB1FE4715.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_2054.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/50000/9AF6AC65-B153-9D41-AD4A-A63A7E8BD702.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_2057.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/50000/77D03341-50E1-8E41-9DCB-7143F08ED748.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_2056.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/50000/B3250522-3CB5-784D-888E-A1BC48689C4D.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_2059.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/50000/92FEC7B3-DCE9-384B-864F-D9F3BDC2A11E.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_2058.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260001/B408FECE-DC1E-ED46-B60D-5030BCF48448.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_654.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260001/AB9B69D4-6227-D649-9A5A-599AAD14D189.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_655.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260001/168B77A4-1126-2048-BAB1-5DF5C398306A.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_652.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260001/2D266BE2-CFEC-CD49-804C-941C1C578473.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_653.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260001/7B19DE5C-AE6B-4940-A789-F0B9FBF46722.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_650.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260001/62862446-DCB2-0A44-8AED-0D6A2A0D89F3.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_651.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/40000/8B7C10DA-37D9-BA45-A189-513ACB9A51C6.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_1364.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/50000/83990764-5794-2E48-A495-0734F4F3FA6A.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_1977.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/40000/C050A709-9034-2846-890D-E3B1E2BD158F.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_2699.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/50000/5287A182-09CA-3D40-B73C-1A320D3A8081.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_1976.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/40000/AF4CA30F-00A3-2C44-BD18-52019464670A.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_2697.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/40000/6A5B96D3-87F9-764F-86EF-5E662180A922.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_2696.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/40000/E72C73F6-BE7E-854D-A5B8-D0F2A6EF6A1A.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_2695.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/40000/BC87312F-638C-614A-9500-19FC77B10F81.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_2694.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/40000/39337A47-335A-4B49-A164-87CF677C53D1.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_2693.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/50000/BF232D47-0B5B-CA46-9C62-A99D0B433B2B.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_1975.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/40000/BB7AB88D-6E4D-F94B-803D-18D6D204F8F3.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_2691.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/40000/D6754DAE-BD8A-A743-BFCD-2884943FE578.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_2690.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/50000/9911530F-7125-7F44-B3F6-2A931A34A682.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_1974.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/50000/88CE2636-D69D-9E4C-A4A1-4A1827A00DB4.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_1973.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/50000/92EF6EC4-0ABA-BB4E-BEB9-75EC901E05F2.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_1972.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/50000/1F044773-F226-8D45-8254-CC4ACC262DB4.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_1971.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/50000/ACB9AE7E-2404-F246-8079-836D332F0610.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_1970.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/270000/CAE5A00D-F3CF-1147-87C9-5CE8A0A8502D.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_1362.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/50000/58B97CCD-355F-FB47-B5AF-257CF72B3A5F.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_1968.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/50000/B6B6D7D0-07B5-2A4E-87F4-1F4644BFBE07.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_1969.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260000/A11FC3F4-03D4-8242-81D9-61EF25050BE2.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_1618.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260000/CB84D58B-8ECC-3648-A28B-5CE1C9956D37.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_1619.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260000/166E1B31-53CA-7C4A-A621-7B4E4EA6FC7E.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_1616.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260000/D9A96369-3351-454D-A90E-CB8188226DA1.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_1617.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260000/A4615C03-51ED-2140-9215-6A77B1B58C3D.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_1614.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260000/10B8B995-E831-6B44-9569-2A9B429A2431.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_1615.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/50000/91753DA6-5BA4-C242-91D3-BC0D0D61BC77.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_1960.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260000/9F2E0339-6FDB-3744-BC9F-119E842F141D.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_1613.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/40000/FA6D8E55-1D8A-DA4A-81AA-665BEC42B8C9.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_1610.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/40000/3ABAE2A1-3A76-054F-BF8E-C39E84D530E9.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_1611.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/40000/F3FF1F15-E0ED-924D-96A1-56806E4D32A1.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_1363.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/50000/FF05ED2D-24FD-9A46-96C2-D6FB42DDEF7D.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_1768.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/50000/9286C979-0EDC-C249-9970-83BB3C6175E3.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_1769.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/50000/26BDE0CA-5C94-E04C-97E3-29F30C27A6D7.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_1762.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/50000/38948ADE-EAB8-2742-87EC-0023B2504A21.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_1763.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/50000/05953028-0B21-EC4A-BA1D-E55E9C1A2284.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_1760.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/50000/723BE6A8-B750-AD42-B9BF-C4797A0A88D7.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_1761.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/50000/FF5B362D-EBFE-C243-AECA-FE451D019647.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_1766.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/50000/2414B8A2-CC6E-7E4E-9DAE-28CB66DF7171.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_1767.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/50000/DFB83047-9FDE-2444-BD9E-4B6A37EBF190.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_1764.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/50000/AC8CC25B-6FC0-7643-ABDF-8E6E01B7CC44.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_1765.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260000/A54CE278-B167-774C-8AFA-AB1B73E62B7B.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_1142.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260000/271CE970-6E4A-8146-926A-F3E2D987F725.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_1143.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260000/D4151183-F666-E44D-A343-2A6994D829E5.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_1140.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260000/AD0C8AB4-4D5E-D54A-AB97-C9E65EF82D37.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_1141.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260000/77F8D834-CBEF-A045-AF23-68D838B7B415.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_1146.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260000/43222D5C-0ACC-3149-9721-D0127E92CE73.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_1147.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260000/A3978086-D477-C144-95C2-9803017764B9.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_1144.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260000/D55375E7-3DB2-6D46-B6A2-CFD757AEB7D1.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_1145.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260000/6A2AE716-2CE2-D54F-AF1C-8704C83C741E.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_1148.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260000/DB351221-25B1-0F41-9473-54BFD9739F11.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_1149.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260000/76E49DD7-1F1C-F34B-BD92-0B269D0C648F.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_769.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260001/38F76B04-1E28-4942-8046-FA8F20330F5A.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_692.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260001/5500D2CB-340D-C445-ACDC-DEAD1DCD6F9F.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_693.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/270000/84C5CCA4-5A33-B24E-852C-AD063A506297.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_1544.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/270000/FF0F4082-A40F-0041-929B-A0E086E02FBF.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_1545.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260000/C9E01D3B-B24F-4D42-ABA2-23885DD09B21.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_696.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260000/D6C902B1-B08B-5049-A880-69733DEACCB6.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_697.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260001/65CDBAD1-F80C-264E-A46C-FF52F75A1E0E.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_694.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260000/EADC8C11-D4EC-4841-8180-CF0A7ABA3125.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_695.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260001/A6586D89-91F2-BE44-BE61-A2951105A033.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_698.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260001/11E01775-7994-F949-B326-E62947D24325.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_699.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/270000/2C23D933-BB93-654A-8578-FCEA2181CF8C.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_1548.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/270000/8BC5ABBA-F933-FF4B-8EEE-18E34CEE3239.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_1549.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260001/061A612D-F3C5-CE4A-BBDF-F2188F0381E7.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_542.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260001/B2CB137E-B97B-0C48-BC9B-0BDD415FE1D6.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_543.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260001/C3444593-933F-6045-BFBC-D91719ECCC9A.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_540.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260001/B745AB91-050D-E046-9379-38557402970F.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_541.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260001/6B97F989-D0B0-EA47-85DC-CD2A9B80C621.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_546.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260000/358CD339-36AA-4541-99BD-D23F4041ECD0.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_547.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260001/B3DB9ADC-CA0A-FD4C-90D4-05326A3921B3.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_544.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260001/D16EDEA8-94C5-814D-B9A7-6904C1ECB827.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_545.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260000/E790A5FE-149D-C143-9354-3DA57E4B9BFE.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_548.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260000/3FA82289-3C10-4842-961B-44A61B177FE5.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_549.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260001/C0C39C80-2CC2-C440-B7DD-7F8C876B1FD1.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_760.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260001/53A81429-34B5-7348-B473-3E0D72960385.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_761.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260001/76095365-C2DE-B74D-B6B9-6497802A1BA6.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_766.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260000/65EAE479-F204-1A4F-B5D0-FB53F981219F.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_767.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260001/7585DF20-D993-8E4C-9373-8A52A1E51235.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_764.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260001/E3AEC93C-04D3-164A-9710-4CA6C3AE6DFE.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_765.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/280000/0DAD7BB2-EAE2-5945-BE26-FEAC5A482E02.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_414.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/280000/7E7D7EE4-ED29-5448-BB06-F4FA4065EB36.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_415.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/280000/788C6928-B972-6445-9E53-7FD13726EEC3.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_416.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/280000/87F6CA11-E05B-F545-9660-FE049C8D2CAC.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_417.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/280000/A44E7711-D76D-3447-B967-334407FE3783.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_410.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/280000/75226B82-1080-B542-B715-74B6AA652EAF.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_411.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/280000/B1DED86B-D92E-B84F-9E8F-0D7D8B717ABA.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_412.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/280000/274E5E53-5C3C-8049-94AE-FCC16E2A796F.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_413.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/40000/E8479A1B-5514-C841-8CC5-B49FD555AE51.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_1384.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/40000/6AC41834-99FD-8F46-8314-9EA352FE270A.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_1385.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/40000/222B26BC-AE13-A544-9E00-A4A5D2A98E96.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_1386.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/40000/82BAF408-BD47-BA45-B492-7524ED972D05.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_1387.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/280000/EDA5200B-993A-654A-9A00-E563B6FD69BE.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_418.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/280000/99EA938D-9D1F-3E49-A77A-C07864BF6EF8.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_419.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/40000/5B4DB8D3-B079-0046-A819-C18410342EC1.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_1382.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/40000/5AEABAC0-FE03-9C45-A84D-296EF6213EF2.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_1383.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/270000/537AFDA0-3CAA-7B4B-9DEB-526B5468930D.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_1254.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/280000/35956EA9-DFD1-4F43-BFA2-287A3D996816.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_368.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/280000/8A64DACA-BE0F-EA45-9211-763693BA9B8C.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_369.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/280000/E8B25E46-394A-F945-8940-DB3B837C0769.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_366.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/280000/3D3D9F34-BA26-214E-9CAD-D73F63364CFE.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_367.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/280000/4AC7F42B-D817-BF45-B1FC-949102C3C99B.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_364.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/280000/74A38C17-3456-404D-A5F3-131964462A1F.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_365.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/280000/ED1EB978-7B13-AD49-A0FE-2CDB9DBDD9F9.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_362.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/280000/2BC6961A-0837-2A44-83C3-AC90B6232B09.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_363.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/280000/A1E5066F-0AC2-7649-BC25-D38C1154E844.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_360.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/280000/03952CE6-0D1B-504F-8053-24D6B52636F8.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_361.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/280000/B9C4AC53-0B93-2D4C-8715-2087FD5835B4.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_380.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/280000/BD42283D-652A-E345-A8A0-0899BF53AFFD.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_381.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/280000/14439DED-B42D-4642-A7D0-8BCF1A4555CA.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_382.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/280000/A967B29D-A7C0-BA46-877E-87B06193EE1A.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_383.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/280000/71B545ED-3EDE-5044-A8D0-693F2EAC984D.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_384.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/280000/F28E190B-F725-5F45-9854-A25F5769E8BA.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_385.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/280000/B5D5C351-D3FB-9848-9A3D-EF845C43E796.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_386.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/280000/D52FECF8-5D05-BD4B-85BC-D3E1F581A3A1.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_387.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/280000/D4C831F6-9851-924B-90E0-42464C6C75EE.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_388.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/280000/8C7DB9AA-EBDC-724B-9DDB-1EC3AB9A1059.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_389.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/270000/5A582C93-9C99-004D-AC62-7ABE40E00D78.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_1253.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/270000/B7D6DCC7-330D-0946-B612-B88BFEC6B466.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_2190.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/270000/BBFB0D8F-5B5D-4D4B-B58D-F9281C8D0D54.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_2191.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/280000/68F064C8-7310-094A-B238-ED2EC5F51B74.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_258.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/280000/5B9FB122-BD5D-5545-8544-857E8542DE56.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_259.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/270000/313334D2-055A-B940-BC3D-6B1E77B248E8.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_2194.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/270000/33B9CE17-8D42-B149-8FDE-27B1FD99D902.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_2195.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/270000/72F22FCF-3C31-0544-A466-B9D796436B85.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_2196.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/270000/060331B6-BE6C-B340-975D-495AFA6D12E4.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_2197.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/270000/959996AF-3A41-9240-A46B-8D29D68720D6.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_2198.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/280001/06139014-8DB4-D547-876E-CB9D2AF77171.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_253.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/280001/B95AE4F2-5007-1C44-B281-116898551EE6.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_250.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/280001/B5ED0B0C-5AFB-5644-B320-40C63C0579B0.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_251.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/280000/6202E7E5-AEFB-9648-B314-4F53ED51F1E5.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_256.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/280000/F46393A5-4F43-5C4A-9731-8D88C0637729.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_257.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/280001/267C3832-7776-DA4C-8C0E-934E36A27354.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_254.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/280000/4EB48680-3FDB-5344-98D6-3D8B52DF4323.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_255.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/270000/568E8983-2DFC-084F-BA2A-32E8CB0CBCA6.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_2200.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/270000/9CD9586B-3524-194F-97D8-5AA8B638960B.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_2201.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/270000/F23912AC-127C-CA4D-97A3-CB6679BE3D36.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_2202.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/270000/DF00961C-923C-2B4B-B205-3D0CCB067454.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_2203.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/270000/F7ADD96F-C144-8340-B012-98242BA9FBA8.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_2204.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/270000/04061E03-25E1-EA40-A0F3-BED251D71565.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_2205.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/270000/453C9B15-6BDE-4B44-A4B5-4CAA770D4FD7.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_2206.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/270000/746D4FA8-EDD1-6242-9B05-B6C7C69AAD43.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_2207.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/270000/235D35AC-DE5D-3F47-842D-E8378D34B6D3.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_2208.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/270000/B09F6BEC-AB64-0345-9967-A96AB9D6B9A8.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_2209.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/280001/D5B6A96A-27E9-2441-B7D9-C816C684C6BA.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_2428.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/280001/F9ABA209-631E-8646-B65E-1D802EB17728.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_2429.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/280001/30D9D18B-7A22-6C4B-8DF3-C9B84820E55D.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_2420.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/280001/CEDB4A62-0D80-034C-834E-A387FCDB34F1.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_2421.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/280001/68BBCEA4-91C9-A14B-AF26-060CECA0C251.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_2422.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/280001/65041187-4075-0D48-B1FB-9E2ABF115ED8.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_2423.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/280001/3B05E99A-3F5B-3548-8733-0C629C9C9779.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_2424.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260000/0715EFFE-09C9-7347-B37E-CE5CCE65B80A.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_900.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/280001/343E47B5-D87E-1C43-BC12-E8360111445B.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_2426.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/280001/584A18AB-B4FC-A049-AD0F-51765399AC4B.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_2427.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/280000/D18D8DE7-7D2E-254A-AB97-2E4966F0433D.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_2432.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260000/7A5E06ED-0146-B749-8AEC-7C41ED00B9B2.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_901.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/280000/B1BD188B-524D-DA41-BD71-8D7E659B9138.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_168.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/280000/39DC0D1D-4923-A741-96DE-337FFB7D2627.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_169.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/280000/5EC3B5FC-EFFA-7B4C-91C4-9B263C2C50C8.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_164.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/280000/FBE46181-F48C-AB44-8DAF-E2A2BB0A91CB.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_165.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/280000/EEFAB0FE-9071-FA4F-B7F7-EF3D45AC4343.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_166.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/280000/2DC32A36-1222-304A-9A62-27CB8D484BB1.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_167.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/280000/8A3BE165-F89F-484D-B13E-83F78F3826F8.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_160.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/280000/CF565F92-5EBD-0040-9BAC-A325A1AD942F.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_161.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/280000/00907B46-CB2F-1744-A132-69DD2C2B07CC.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_162.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/280000/B57DDD46-6AFB-3843-B581-8BFA81745B6E.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_163.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/40000/A182DBE0-927A-B744-A034-0E70887EA0AC.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_2518.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/40000/5B5750A3-79E6-8549-AA5E-5512A0B1C79D.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_2519.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/40000/5865259B-213C-BA46-9B35-6CF7F8BC1A3F.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_2514.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/40000/BCDB1DBA-D3A6-F040-8184-618EE09EFFA1.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_2515.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/40000/546296D5-8882-3949-B7C5-7228797E0A40.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_2516.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/40000/51C9CCE6-E0BD-4B41-9C87-03EEF4AC2C08.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_2517.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/280001/132E51AE-6E49-DE42-8C2E-2E55C9B29AEE.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_2510.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/280001/E8E1DDF4-A63C-7848-A54C-289B8A4E3B3E.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_2511.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/40000/2CAFC2D1-6F0C-0F49-B02C-400E4276BC48.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_2512.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/40000/D65C02E7-0A80-314E-B804-494838B84817.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_2513.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/40000/C6EDC6E4-AB3F-7347-8F5E-4FF98E8C0954.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_2734.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/40000/BDB499DC-4FC0-7D44-8F89-9607D3CEDFFF.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_2735.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/40000/FBEBDE3D-140D-494B-A511-F4B4185B0144.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_2736.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/40000/842DAEBB-87B7-AB4E-9412-3123F37811AA.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_2737.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/40000/695DAB55-9AA6-724B-ADFA-5C0C5032269B.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_2730.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/40000/627DFBF1-D517-2E43-9183-D27AEB491542.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_2731.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/40000/9163A445-83DE-6C4D-A23B-DC7D3357FA78.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_2732.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/40000/6292837C-8054-944D-B35A-D5C3B0D309FD.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_2733.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/40000/B3CCDD70-0211-8B40-9EEE-DC7902C09483.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_2738.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260000/B794B036-4286-1447-B3BC-31B2E2386DA5.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_1090.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260000/B432290D-5748-324B-95FD-282CA9DA2F30.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_1091.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/50000/3DEF64A4-FBAB-BF46-B76B-9BD202C85706.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_1814.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/50000/77DEFDC4-2A9B-E64C-94E3-72316FCD5B45.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_1815.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/50000/416D1966-BC0F-5C45-BAC0-BC7E029CE812.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_1816.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/50000/E4E8AD1D-DF3E-8641-8657-36410CDFAD52.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_1817.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/50000/A3C60AED-DBB8-DE4A-9B94-59ECE1D729F8.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_1810.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/50000/9F6764CB-19D6-0743-B5F5-56815C496BE0.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_1811.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/50000/FE188E10-1572-8B47-8CBD-BAAD5BCC8F73.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_1812.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/50000/4A665E0F-0AD8-FC49-987E-36568481C9BA.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_1813.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/50000/6DD9719E-8F86-294D-AEAC-867D2E950939.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_1818.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/50000/3693DBB3-4942-3840-A31B-127346A26A92.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_1819.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260000/06891533-FC18-7646-AAEF-5707CD0C303C.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_1098.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260000/157C2781-50B7-3848-91F4-4BB79C6FB2AB.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_1099.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/40000/B0622AD2-B59C-5B4D-8E21-2D7F84369C97.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_1609.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/40000/6DCC44F2-F91E-0A42-94CB-E2D14794AF17.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_1608.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/50000/E63B635B-CCD9-8E4E-B47A-3A56B97BE274.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_1979.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/50000/5342AE8C-2A0F-C240-AE49-09BD85846541.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_1978.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/40000/59B90254-F5AC-144B-8B3D-614913655169.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_1601.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/40000/3A717D24-C9D5-A34F-A07D-B4C2081A59E9.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_1600.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/40000/B1736C18-744C-0E45-949F-A59F54900440.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_1603.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/40000/3604A816-817D-4D48-81D6-7515C039A92D.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_1602.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/40000/C5DC574E-878C-2340-80C5-956EF203495D.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_1605.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/40000/ABC5212D-48E1-7946-BDB3-4CFCA38C150F.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_1604.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/40000/198913F0-03D6-274B-8A51-9AA69EF07908.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_1607.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/40000/C52284F1-DAD2-494A-8E04-5E8A66A5C176.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_1606.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260000/814DA621-8793-D343-AA2E-10A3B766CBA4.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_809.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260000/B79EE08B-E0AD-3345-B906-6F7941CA122F.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_808.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260000/625E4177-B902-434A-86B9-6EAB9456A5E6.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_803.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260001/DFA165BF-3D5D-6B47-BD92-7C9960B3C553.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_802.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260001/C3ED02D2-F89D-3440-A27D-56DFBE3D7299.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_801.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260001/E5B9F1F9-A0A3-1C42-A2A8-7B370807CF2E.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_800.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260000/8D90D194-13F5-D44D-AAC9-C07CFFD4F42D.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_807.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260000/027298C8-F7C2-7848-B241-DD6F07F41C2F.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_806.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260000/60F6C598-3743-9142-99D1-F3BAD38CDE23.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_805.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260000/A2EF669C-1FE3-EA4D-8166-6B0A5D4DEFF9.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_804.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260000/9CD97F09-4021-DC48-9A71-23C2ECA8A000.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_909.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/50000/070D0B22-4990-764A-97E6-F79AABBE9C94.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_1775.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/50000/8DAEC55D-CDA8-FE43-8510-72D35DCD2C98.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_1774.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/50000/089E93A8-56C4-C142-8BDF-66F7CA1E1874.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_1777.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/50000/25F0D22B-79BF-FF46-93BC-1C5B1D420868.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_1776.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/50000/E4FAAB51-AF72-5943-A0EB-9FE79FA9C74B.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_1771.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/50000/7E081D7A-D361-C542-B9B0-983C3C3CAE29.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_1770.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/50000/FB44020A-3DEB-E940-8065-500FD0199F7C.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_1773.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/50000/71FDE993-68F2-2F45-9A1C-F44966FC231C.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_1772.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/50000/4C89A698-6DE5-4C4B-8A7B-4EA2C449D237.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_1779.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/50000/0EF74DF3-BB79-E042-81BD-B29449F149DD.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_1778.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260000/668AB6C3-26F7-894C-B480-5F36A2A743FF.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_1159.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260000/AB9D3A3F-31BD-854C-A38A-EB51BC596B5A.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_1158.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260000/84C20EE1-F7B2-1C49-93C8-06402DEFBD16.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_1155.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260000/D17902B2-D156-F448-B44D-DEE7DCC1BB9D.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_1154.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260000/999EE684-8C5E-7E4A-A911-64D5F7CAB975.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_1157.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260000/B20AAB06-0126-9D4C-BBE8-BFC418DFFA88.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_1156.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260000/3991274C-2046-8148-AC93-19D8794A4128.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_1151.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260000/D7D618DA-EBE5-984A-AF54-69AA191652A1.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_1150.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260000/9639E3C8-5902-DC47-A862-82EBA1253452.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_1153.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260000/1D7B1D25-4C6A-3443-9701-F0751EB9AE79.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_1152.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/270000/A7FB9471-A904-B547-8F97-36654C166B9B.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_1555.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/270000/4F763C22-F4E6-994D-8B65-709521725AAF.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_1554.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/270000/65D3230B-8B97-4647-9658-1DAC0E9F28E2.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_1551.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/270000/A41A82B9-9536-D648-AF29-430DC669A752.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_1550.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/270000/6E6DA5F9-84B1-7D41-AED0-74D2F89E184D.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_1553.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/270000/5C208A7B-2707-0B47-9219-D0E15CFC0F5C.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_1552.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/280000/D2C83EBA-F893-4848-ACBF-250E31D6E0DB.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_59.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/280000/D7F697A2-A70A-0042-A2DE-FA4D85ADE07B.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_58.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/270000/982625B0-FB86-B946-9190-EA136B9DA48D.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_1557.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/270000/C7726F9E-857A-AF4F-BF9B-742533F094FB.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_1556.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/280000/F8790A57-85CD-1048-A3CD-82DAE105A385.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_55.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/280000/40ECF668-4E4D-7746-B6A4-786E3EE7CDA0.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_54.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/280000/B38E40C1-B16A-A24B-A266-BEC16506744F.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_57.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/280000/D138303F-891D-0549-9454-E04011D6CFB8.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_56.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/280000/DC8F9AA4-45E8-3C4C-BA68-2A4684698FCC.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_51.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/280000/5BE70358-FA62-D84C-9192-1993707EB58B.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_50.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/280000/B15B1021-B54E-4D44-A083-A30EAC4B1738.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_53.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/280000/5E638396-217D-AA48-9AA3-A68E2AC8A2B0.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_52.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260001/E7B1823B-83A0-CC40-86B4-C0EBA1DFB4C0.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_537.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260000/6289ED40-E9C8-064E-AF4D-AEC4F70CF4EE.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_536.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260000/C71506FF-1860-CF43-8BAB-3FFEDBE81004.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_535.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260000/698F6901-7964-A740-84CE-2E9271CC8D87.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_534.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260000/B437302D-03A6-7144-82D5-409E795FD124.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_533.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260000/FA720C61-4D95-C046-9A21-3B6C6BC80AEE.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_532.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260000/7B8496DF-0AC9-0A4C-8AE6-5CF0595815D3.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_531.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260000/C73166AE-78EF-1C4A-8DD9-DAE00F14288B.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_530.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260001/56A2DE55-642C-E64A-8EE6-62ABE76682C7.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_539.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260001/45F6F233-FD6E-D64E-8946-6422C3B6FADA.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_538.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/270000/E5FB5753-7B6A-C746-8836-FDA100F78A07.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_1558.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/280000/06987D52-C363-C44A-A41B-CFE71AABD847.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_429.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/280000/0EADF20F-A58A-304F-BE63-60FC93BC6EFF.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_428.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260000/FFF0FBD4-328E-CF4A-B38F-F198B5FEB322.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_1399.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260000/5DF9CDF8-3FA2-C047-AC21-D83701B005B0.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_1398.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/40000/C662C1A8-82FA-4A43-A382-2B3CF72A3087.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_1397.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/40000/EF4CDB7C-416C-6B44-8AA2-891548F78E11.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_1396.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/40000/B5C6153E-53EC-964C-8900-A32886AD7682.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_1395.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/40000/D3AB7152-B877-424F-ABAC-756312EB52D8.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_1394.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/40000/4EFE6435-93D8-F04D-BA9F-D53F23EAEED6.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_1393.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/40000/C9F46C02-69D5-9D40-A5CC-0B7E155A68CF.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_1392.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/40000/82AD45BE-238F-F34C-815F-65E45D4ABD04.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_1391.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/40000/7667C626-9A89-054D-A746-1569397E7D1E.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_1390.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260000/6F5758CD-844A-FE4E-BE0F-D7286CA384B6.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_1420.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260000/B97B4E77-D899-6946-9766-95902B5FB912.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_1423.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/270000/DFE359DF-882A-0B42-936E-F38ACBE30676.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_1443.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/270000/50341962-D7A7-A147-9C06-E56598087842.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_2183.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/270000/66032601-4B5C-BC46-A97F-3FE67E2C4C0C.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_2182.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/270000/B5BDC567-1BEB-B745-9DFA-46453C1191DE.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_2181.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/270000/E50C95BC-87F6-1A4D-AFA8-83D1B245D2BF.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_2180.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/270000/53358215-61DF-2744-A3B3-1B593680F82F.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_2187.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/270000/A95855A6-4489-6249-BA23-D978C8B77774.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_2186.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/270000/85A44D18-594C-574D-AF53-9AEAE58AC13A.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_2185.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/270000/EC68AFCB-7630-B44E-9693-72B330F037E0.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_2184.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/280001/D19BF8EA-4F16-BB4C-89D2-06FAD685BEE0.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_227.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/280001/F4D950A0-2A3D-3A41-A7BE-E67CB62C674D.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_226.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/280000/47C2447A-28F5-7245-86E6-28B17E26F4B3.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_225.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/270000/F489ACEF-0C59-A749-A30F-4F4A794E5395.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_2188.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/280000/AFB2B812-D725-6242-8778-0568F8D59708.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_223.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/280000/4EC06BCD-5ED5-3F42-AD33-87B4DEA8479F.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_222.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/280000/ABFF15B2-DA2E-B94F-B284-99BC57C10921.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_221.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/280000/B0F0A860-E34E-2F41-9AE3-DB7D5F8FFE08.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_220.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/270000/A715807C-A47F-FA4D-8A75-683018FA4922.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_2213.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/270000/6DC60191-61AD-CF47-85CE-1E6B07178CE2.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_2212.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/270000/35C3C86E-91B2-234D-9DFF-DD1B6B1AEBF1.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_2211.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/270000/E756EF23-2320-9546-B093-457653F9C875.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_2210.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/270000/38B9EF69-CFA4-8F4F-97B9-23AAEE452396.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_2217.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/270000/CACDFAAF-2ADB-8847-A90F-6B48AE7FE590.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_2216.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/270000/C5FED718-0D47-074C-8B1C-5FF87F6C7E20.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_2215.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/270000/A4E724B3-A2D6-3C4D-A759-A82F87E85A8E.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_2214.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/270000/C0E68692-5905-754E-A12B-A96F54CA5CAD.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_2219.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/270000/DAD6FF42-AB5C-4546-B621-A69DFC487B79.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_2218.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/280000/BB97D843-7875-EA4D-B4E9-7956984D2256.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_151.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/280000/36C2BD5F-E49F-264A-A5F5-02CCC956E7DE.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_150.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/280001/50A381E4-8A0B-0A47-95FF-015ED33C6892.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_2439.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/280000/045FE889-5BCD-FA41-A198-B5B49E59B937.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_152.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/280000/054FCA32-40EA-1F49-B751-419C93BF8734.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_155.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/280000/58166770-7527-B64D-89DA-2251488800A5.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_154.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/280000/1B1C0F6F-5E9E-954B-9038-0C10E4E9E8E6.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_157.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/280000/3EF1957A-6959-7F4D-A12F-E6780C4151E5.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_156.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/280001/7EC9A76B-65DE-2B44-AFAF-327FB80F953D.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_2433.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/280000/1E68D015-B7C7-054C-B091-A01EF765AEE2.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_158.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/280001/A65ED1A1-07BE-0048-86F2-D66206D9CA0B.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_2431.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/280001/77A7DE56-CE96-3247-B9E7-38DA0940EBC5.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_2430.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/280001/0CF4BE92-6702-2B42-A42A-DEB66E8AA8EB.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_2437.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/280001/EB7997AF-F3DE-2A48-8175-13D838C392FE.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_2436.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/280001/AF69BB35-58B0-D245-8AF1-C5C862BBE41D.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_2435.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/280001/FDA83304-4EBC-5343-A569-A3BDBEE6DFCC.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_2434.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/280000/C336E688-7D7A-BE4C-A5DB-50DF6FF140D0.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_123.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/280001/9D8661F4-693E-E943-84B4-BAD0DDFE93F8.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_2509.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/280001/C9B1CB1B-8D9B-9B44-BBE4-50A605D6A1C0.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_2508.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/280001/E11F891A-0AD6-054C-B82D-299DC7DD5F27.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_2507.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/280001/EED7837D-C831-7342-AF20-F45472F665BD.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_2506.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/280001/0C6690F6-FF17-5646-AB7E-DB293EE4E158.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_2505.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/280001/FBAE309D-9341-B647-B4B9-0503DE339A43.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_2504.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/280001/1CBFD0F1-1BCD-C643-9844-F169433BA4A1.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_2503.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/280001/FFF5D231-776D-2647-BD63-060C9A94ECF7.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_2502.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/280001/7CCC581B-975A-2D4B-ACD5-6EEC63DA3DB1.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_2501.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/280000/F8A2CCC6-8891-D848-9485-65171AD36944.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_2500.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/40000/DC3223D3-A2A1-E34B-A746-80C9C5A5C687.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_2727.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/40000/309F6769-76D5-D64F-8AF8-4AFAAF44C7CA.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_2726.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/40000/D111FA93-5922-AD44-A12A-F8EE52C31A2B.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_2725.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/40000/9882E1A7-BF2C-B34B-9923-A42A09BA40B1.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_2724.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/40000/87142E50-BDAA-B249-9201-0A295B8964DB.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_2723.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/40000/38DA7EA8-DA7C-6746-948D-8AD5B4222E4B.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_2722.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/40000/F9920DB9-3AE4-3F4C-AADE-5C88B6159A1F.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_2721.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/40000/31C4605E-8104-5040-859A-19139D25170C.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_2720.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/40000/60F518B7-A627-4548-905D-D4E6A896A4F6.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_2729.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/40000/23A0A3F7-D4B6-6D44-9D69-591742BF1B8C.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_2728.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/40000/612C20C8-D59A-8C4B-AD62-95784C139782.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_1712.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/50000/A57B10AE-C778-6D43-9E0F-D27A63E33320.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_1807.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/50000/BCA88D1F-9D2C-3943-84C0-1557B14C7840.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_1806.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/50000/33E97142-4299-7B42-918D-342231E9178A.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_1805.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/50000/B3BD755F-D6AC-8246-BC3C-2E0282BBE7D6.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_1804.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/50000/B6848429-CBF9-1740-922A-FB39C7DF94F8.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_1803.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/50000/450CC65B-58A8-A641-8689-C76FCA06151E.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_1802.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/50000/BD1C7530-2439-7F48-BE9F-CBC7D88010B3.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_1801.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/50000/83709367-F717-BD45-B8BB-219FC3FA47C5.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_1800.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/50000/355032EF-BD83-9F46-9D44-847F86A92104.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_1809.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/50000/8DA366A6-8D08-BB42-A6B1-FC3D940A0A73.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_1808.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/270000/5FC82B74-3990-CD4A-8779-FF028F1BD299.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_1256.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/280001/ADF8AE50-682E-B04E-9298-DAAE13EEE7B8.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_1948.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/280001/A172688A-5AF6-B648-9A44-35D8FD51DF38.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_1949.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/270000/39A19709-7C65-904E-BB2E-FCBE254611DF.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_1257.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/280001/C09E9E19-7640-1C4F-BBFB-0E562CDC2B64.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_1942.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/280001/26DEACF7-58A4-BC46-9B92-8798C56E3ED0.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_1943.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/280001/2F92B9E2-843C-2D4A-9DF6-F9F198CF705A.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_1940.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/280001/8BDE4771-D859-3B40-9800-14EB80CFF3CF.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_1941.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/280001/90D05FAF-9B08-964F-A7BB-5CDBC9922D8A.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_1946.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/280001/D8B1F9E5-433E-2542-B979-5AD96C0B5ADB.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_1947.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/280001/0B20FC8B-03DF-DE4D-835E-5E479D1AE7F9.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_1944.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/280001/A334AF4E-455D-534F-AF0D-52F68BEB78A7.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_1945.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260000/C9AB3583-8C4E-F24D-871B-E12396629FA7.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_818.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/280000/EAEA0373-15DC-8E4E-8DA5-99DA41FE71BA.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_0.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/270000/A26E2515-464A-314A-8417-AD56905A3ADD.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_1255.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/280001/6299542C-1007-0046-9836-A363FB05A507.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_2425.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260000/09181BBE-2688-BF45-9210-E5272F847646.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_810.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260000/749AE8FD-DA93-464D-9D25-225872455317.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_811.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260000/3A19ED41-9758-C647-BB4D-2A0E03BB48E5.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_812.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260000/DCE41BCF-257E-294A-8B78-C431DC1CB3E1.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_813.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260000/EB12A5BF-4225-D741-A409-E5EE551F3C41.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_814.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260000/41AFE377-0D01-8C4B-9890-9B9742438062.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_815.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260000/9E982D43-E502-AA4B-9C68-BBA0FDC0E4C0.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_816.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260000/21BAC838-1210-134B-9A2E-E892131D3449.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_817.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/50000/697F5B36-8B42-B047-AB7E-D27A31688EF0.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_1991.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/270000/682336BE-7D4A-1D44-AC22-90D58B3CEE30.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_1250.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260000/0C9F9B98-3AE4-4F46-A1B6-59BAAC2A0A34.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_1622.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/270000/8E7C8B75-B08F-E040-B5C2-DB9ECBA118E5.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_1251.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260000/666C4ADC-84E7-3443-A511-7B24AA1A06B7.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_1621.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/40000/517D5D5F-CC03-8445-AB0D-215045404AB9.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_1490.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/280000/536B5642-D20A-8144-A594-5540F00354AD.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_421.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/40000/69E24232-269F-EB4E-AFD3-49D7A56418E1.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_1492.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/40000/7879B130-86EB-944C-8CCD-1521A83711ED.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_1493.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/40000/2DB26028-EE3C-6341-948E-5BF93C7156F6.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_1494.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/40000/C920AC2D-9F94-794F-A645-7A3DF849293C.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_1495.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/40000/A2EB74B6-6068-1A45-820C-406C21871EB0.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_1496.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/280000/43F43AA0-8AF7-6645-B2A0-3CE5D6D5D081.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_420.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/40000/C77BB363-32C1-FE4F-B85F-778450C47DF0.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_1498.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/40000/214DFE00-D7D0-DC43-ACB9-0AB5E557F66A.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_1499.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/280000/B06D86D3-C543-B643-9B2E-0C4A3D94A2D2.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_423.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260000/8244CFCC-8135-2842-B752-0F8D24121932.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_1626.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/40000/4300FC4E-21F1-3041-B915-AA4B0AB375F2.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_1700.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/40000/27F5E5EE-79DB-A34C-B9C1-FE8DCB9575C3.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_1701.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/40000/7E4345A1-1698-BE43-94C6-D363A045D8CF.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_1702.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/280000/C172CBBC-B73F-0747-A7B8-4B980A6EB003.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_422.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/40000/E8C0A8D5-E829-1841-8107-E540B9ED2A62.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_1704.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/40000/93C0A225-7118-4349-9A78-7A08ED8B8791.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_1705.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/40000/68B68CDD-0D55-3043-873A-344E19D76613.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_1706.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/40000/B912DFF8-FCCD-0F49-8A60-EE12B8BFCC92.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_1707.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/40000/9E15B7DF-BE58-F548-B3E7-48DB8CC52E43.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_1708.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/280000/C5FFD056-523A-B84C-B2FB-A5F69C11B454.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_425.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/50000/88B63B80-CE13-E34F-BD0E-B1D6CA5437D9.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_1996.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/280000/264D609A-C70F-9A49-B7E0-AD898138498B.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_424.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/280000/255B4A63-E2B9-D64A-A1F6-09190AD421DF.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_427.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/280000/6695132A-1C23-A642-AD0B-981657DCAB2B.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_426.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260000/F374E927-0609-F547-8CA9-C9B6AAEF2404.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_1128.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260000/DC19D332-A678-0245-9658-6DE39F493BDF.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_1129.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260000/5871DBE7-70E4-DC4D-BB68-B373F1527CB6.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_1120.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260000/37377ECB-361E-FB47-8AFA-ECB00C2F6400.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_1121.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260000/098AF77C-D5BB-4348-AAFB-E186E1730D28.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_1122.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260000/AA7A81AE-8395-A546-9C8A-D31F41838E81.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_1123.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260000/A81C85EB-C59A-2245-B0FF-C91D218616BE.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_1124.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260000/CE6DEF1F-5CD3-404D-9BFC-B210FEA6451D.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_1125.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260000/C97D5283-AA84-664B-8435-29511837A555.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_1126.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260000/58324CFD-66FC-0746-B982-9E46F16D17E7.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_1127.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260001/7003EFDB-3909-E447-85E4-062C9B4B92CB.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_524.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260001/197613CC-43D6-F043-97F3-F02317C4E835.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_525.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260001/00367AED-61BA-9D44-8037-98A7E293245E.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_526.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260001/5029BC08-FDE0-CF49-AF85-E542AEC75A4C.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_527.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260000/519452D7-C22D-EA4E-A8D8-344FD01B1D28.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_520.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260000/C11DA805-DEAB-CC40-B637-54FEED987300.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_521.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260001/9014538A-681F-394C-B78A-DB1CCC289919.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_522.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260001/062ABB5E-F590-7B4A-8561-5BEFA7F4AA2B.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_523.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260000/1D45E39D-6A0B-B846-8D76-F4735196F38D.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_1014.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260000/4B8AA12D-07FD-8D45-AEEF-DE017C8EF14D.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_1015.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260000/3A6F7716-439E-1F40-B0B1-138293E42315.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_1016.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260000/D3DE272F-237E-1746-A4BF-6AEC026EEFBA.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_1017.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260001/6B8BD188-0A6B-DD4E-ACEC-E9C8F3F88DCB.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_528.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260001/D96F22FE-F361-F442-AB9E-7986D45447A5.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_529.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260000/87895B10-94F8-014F-83C5-3E623A73047E.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_1012.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260000/E9A4A433-BE50-C840-B13B-B96902752887.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_1013.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260000/520ADE77-7953-BF46-BDFF-F939C53088C2.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_1234.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260000/DC39CB9B-4E47-D34C-82AC-5B444CDEAC80.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_1235.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260000/C896ED85-AAF4-E34C-B403-CFE0DF1E3C60.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_1236.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260000/FDA674AE-0A56-124D-A8D6-7C6BD0F3280A.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_1237.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260000/D199C7D8-B12F-0843-A4BF-97FAD8932CA8.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_1230.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260000/375E8354-AA09-B240-AC97-7539A64F00A5.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_1231.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260000/305E1A3E-A80B-BF47-8746-3D735DF6B701.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_1232.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260000/C2939AF3-1A0C-0D48-ACD9-7BAD633180E1.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_1233.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260000/95924ACF-2D9E-C84F-9B27-A01C2E376CD3.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_1238.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260000/4E6A89CC-4558-A64A-9000-864EB682D2BA.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_1239.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/280000/2A9345F6-C1FC-4243-A590-D8BA6183C7F4.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_438.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/280000/5580807F-EADD-454A-9FD8-C1C7800BEE0C.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_439.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/280000/D232DB2E-33F6-0E41-A519-E7EF099AAD95.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_436.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/280000/00640C3C-DE9A-DC48-B80B-3F426BE53074.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_437.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/280000/F8A4C539-F326-054C-BC62-CF5CD1F587C0.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_434.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/280000/876C73E1-A6FC-2C4B-8312-738C03FB4497.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_435.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/280000/43F711E7-355B-FB49-ADF6-B553F277602A.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_432.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/280000/31557433-5500-7449-8060-7F0EC8B5D272.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_433.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/280000/85542FAB-EEFA-1E47-8BE0-A9FC4548958F.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_430.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/280000/17DD4A68-248D-FC42-B41C-8A0CAD0DFA7C.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_431.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260000/0173DB9C-3E46-4741-B80B-9671E9E917CA.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_1630.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260000/7148F244-02F2-7F48-9336-A2DC740CC461.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_1162.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/50000/17C1B23C-CA5A-7340-AE14-8FDA4E84F62E.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_1961.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260000/CD0E9638-75C9-5843-9B75-72F82CEC6FE8.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_995.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260000/B8FE9651-2A31-6742-92DD-E16D2743A1D8.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_994.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260000/8DA8F58A-1AFF-714B-9D22-4E6D2392676E.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_997.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260000/00B87525-94D1-C741-9B03-00528106D15A.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_996.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/280000/5A730EE5-8F72-CE4C-8AEB-44E6D2FDA6C8.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_238.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/280000/E921281A-5577-F348-A4D7-BD3C75586D13.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_239.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/280000/042DDC49-7F1B-7740-832C-4C2C9571FFD5.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_234.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/280000/9EA86F7A-94D8-A943-A0CE-48D1370CCD6B.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_235.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/280000/40FAA700-2CE5-954A-AB37-B34E3EB2888E.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_236.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/280000/5B476152-36B1-1F4C-9457-97AF77581600.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_237.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/280000/C341A456-7F8A-174F-B0CB-204EF750FD76.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_230.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/280000/AFB0D396-2793-0543-B786-FB4E3C01238E.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_231.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/280000/A7C393B2-7442-C94B-B7A9-5749E9C1C1E1.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_232.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/280000/0CE3F779-E289-954A-A29B-227F5948C4F1.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_233.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260000/151E45A6-6F7B-E74A-A24F-176B23250CAA.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_993.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/50000/347A5460-0E3A-D44A-A00B-E5C1AD8C1B70.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_2228.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/50000/B7B28525-E22E-3B47-809C-FA37141F2AEA.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_2229.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/270000/C61CA68B-9949-1147-B2AF-EA3453E824AC.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_2226.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/50000/F4B2108E-07B7-4E4B-934B-BEC81DB8940C.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_2227.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/270000/62BE28C8-5196-3D48-8FD2-C3DE36FD8D58.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_2224.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/270000/6690F709-DB16-1E4B-B6D1-26A90C9762DB.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_2225.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/270000/02352E36-908A-2248-9F31-467D373EF0C0.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_2222.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/270000/9914FF4C-336A-BD46-A189-53029DB55ADD.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_2223.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/270000/0A9C8135-63CA-8C4C-86A0-8BA600EA44E1.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_2220.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/270000/6A1A0851-15C5-7E46-83DD-C4E6D6DFAAAA.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_2221.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/280000/33B1A099-89D1-DF4E-A4B1-ED3BD915814E.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_1.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/280000/4E06DDA5-9934-1E46-9EA2-D333FB8F9007.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_146.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/280000/8DD2D550-D892-AA43-BA44-B670582C490A.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_147.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/280000/9F6FA2F5-BD32-2D45-939F-66B12154D9A5.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_144.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/280000/F8832004-BE90-E54F-BF2A-DD255C4A703F.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_145.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/280000/0D9AA243-5F50-A24D-ABD2-F428BA3ACA35.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_142.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/280000/7DB3DE43-E29D-C94F-8A21-5CECBAE9E3B6.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_143.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/280000/03B045EA-90FA-9440-B865-036C7F74DFD5.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_140.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/280000/B2CDC568-D4F4-AB4C-BF33-D8F492AECAD7.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_141.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260000/75010B90-C6AF-A94F-919A-9D35072CB7E0.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_999.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/280000/6CCC0584-CDF7-C34F-8D6B-439C402DA28E.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_148.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/280000/385DF672-B811-7848-BFBB-E7B15B1ACB7D.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_149.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/50000/B6B46052-36C7-F044-BBCB-19F5B23EC44C.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_2088.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/50000/89E35DF7-7CB2-D246-872F-8377B6AFEE8A.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_2089.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/50000/D8A07575-A1B7-4F45-A4A8-E7C05DBEB3FE.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_2082.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/50000/492CCC0C-5D21-7349-9D52-A2D5556A6162.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_2083.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/50000/6BF06E35-CB0F-9942-B51D-0586348EF45E.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_2080.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/50000/CFAD96C2-62EA-994F-9CC1-2F8092B01FF6.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_2081.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/50000/EC4D6545-5615-664E-B66A-EC1585C6DBF3.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_2086.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/50000/6D16A695-027C-FB4B-9866-690FDC42754C.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_2087.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/50000/40BA59AB-34C3-E244-A5C6-E58892099453.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_2084.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/50000/E601470C-BABC-AF48-85E9-48D19E9C6478.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_2085.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/40000/420308E8-460D-3D4C-B7F2-B7AC0F3AF6C1.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_2538.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/40000/1D0DAF1D-ADB0-A242-A8E7-0FD9D8D7C8D6.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_2539.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/40000/E34DBBD6-9055-3D47-8F98-E3DD1A335609.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_2532.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/40000/C96137B8-57F1-1C4B-A80C-57922594E90B.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_2533.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/40000/79AFBAA7-82DA-5748-ACCF-2FFDB1387EB4.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_2530.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/40000/FBAA6847-6B40-AC40-A9B6-4BC57142349C.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_2531.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/40000/5894BC6B-FE82-D947-BA56-8BA4FE3A8314.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_2536.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/40000/B8C8F29B-2B47-BE40-B8B8-A710A32AD105.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_2537.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/40000/69DC9EF0-D182-EC40-855A-1F1F02CF709E.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_2534.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/40000/7590575E-9826-F24F-8CB0-B78A2625C563.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_2535.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/40000/D22E497F-CA36-E746-8533-B7340165354B.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_2712.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/40000/7A2AC9B2-9093-044A-BB73-5F5CA5A71A87.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_2713.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/40000/1177A3A9-86FB-8244-8137-153ABF376F90.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_2710.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/40000/5813359B-6830-AF41-9177-3914B086BC4A.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_2711.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/40000/B3622D4F-E62A-4C4A-AF73-C1D45B6D3B3D.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_2716.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/40000/3BCC33C4-4C8E-B846-92EB-FE3032DBF843.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_2717.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/40000/EE901613-8374-4F47-AC01-9F56F11951C5.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_2714.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/40000/76D2D4B1-7FEE-6C4E-9D45-FD506AF721FC.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_2715.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/40000/4CDF62E9-AC43-2842-8DF8-10D4C0BA0D0C.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_2718.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/40000/51AC0594-E893-D947-8D44-408FD9324091.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_2719.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/50000/7E8D12DF-1247-B046-8798-324F7A7F6ECA.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_1832.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/50000/8A2A08F1-1C25-864A-8AE0-B1DE9277DC96.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_1833.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/50000/A2A773B1-2C94-4B42-913D-B7C949506D2A.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_1830.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/50000/F30E90C3-8085-9143-8D1A-87D37C1D0FF8.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_1831.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/50000/EE505DBB-4998-974B-8441-340339153FF3.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_1836.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/50000/A8788B9C-B528-2D4F-A65D-0CBB40B1AF00.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_1837.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/50000/C72F9560-CC63-9E47-BF48-B442C505307E.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_1834.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/50000/A9C763F1-FA72-3246-94ED-F64773CE6AE8.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_1835.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/50000/CBEC2BB5-5425-4D40-BEAF-51AF9374BF4A.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_1838.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/50000/7A1C46D5-3EAB-2144-A2D8-621C6C3C9C20.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_1839.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/280001/22108920-15F5-924F-A952-AE1A0C0C5047.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_2406.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/270000/8A82943B-B888-B64E-8A35-6D09E0189743.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_2155.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/280001/0F168D75-7EA3-A040-B7E9-33AD194443BB.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_2404.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/270000/B47BD6E6-FAED-7945-9B6A-86D52B4E295A.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_2157.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/270000/B29C6F0C-AFD3-014A-88F1-C06DC9B9BFAD.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_2150.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/270000/1958B12C-F8F0-F34F-8069-2C87C35C8C72.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_2151.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/270000/A597EF89-DAC4-F942-B145-CFA9F903D2C3.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_2152.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/270000/90F4B392-A2A1-C74E-BE13-441C554888C4.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_2153.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260000/BB886D06-9380-BD4F-8837-9737033A1CD9.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_933.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260000/074128AA-4BEC-6C45-9652-F6C93CF64D8E.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_932.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260000/12CDA63E-2565-EE4E-976B-3357FB100CD2.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_931.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260000/2CB17BB9-EF3E-F041-B077-8F89B3093C47.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_930.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260000/DF612B5C-ED74-2F4D-AD83-A91ADAA55561.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_937.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260000/A5000B6A-8DD8-1B46-BD9D-C12F221E7FAF.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_936.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260000/F3C5D55C-1623-2F42-9B81-3B10A6855071.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_935.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260000/41E130D4-21C4-9744-9406-045AA739493C.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_934.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/280001/E00C8CB1-1E67-5249-B0E5-947124179ED6.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_1955.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/280001/18F29431-E6C9-B14A-919A-7600FE404E03.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_1954.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/50000/30CD66F7-0E15-484E-ADBA-9C023F23ACB7.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_1957.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/280000/AC6D1F15-2542-784B-A1E5-F88CAC7EEF12.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_1956.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/280001/8183F8F2-676A-FB44-B632-256EA10050D2.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_1951.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/280001/806345E0-5C7B-0542-9CDE-3039D53B2D27.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_1950.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/280000/7BC50BB6-5C9A-674A-B847-2D24EF59ACC0.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_1953.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/280001/009BA9E2-3D08-404E-83EF-856B4D297EE2.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_1952.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/50000/F3F164B6-B8B9-1647-B4E6-C877D96DD9BD.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_1959.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/50000/5CC9FCF6-7247-004A-8872-23588D24A091.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_1958.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260000/E8203CF2-368F-C843-A185-AD9109C215F4.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_829.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260000/9E6350ED-7C27-C948-8441-300B1B5ABFDB.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_828.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260000/C1C62595-AA27-444B-87EE-B0E8105A8DC6.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_825.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260000/2466B5CC-0BD8-3047-8E90-42FE96DE8E2F.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_824.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260000/59F72BA3-5DCC-2C4C-9464-65DEE796BA56.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_827.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260000/1AEA61BC-4DDD-0F4D-AA5A-CDB7EEEDF350.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_826.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260000/6A9705F8-884B-2B40-B8B2-A4CC0EF6D6FF.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_821.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260000/B067AA77-B6CD-F144-AB5D-005C8BC1B879.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_820.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260000/BD2CA945-DF90-DB4E-BAB5-3C405769FAE0.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_823.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260000/3A65BCC7-A1C1-7E41-888D-218852F9A305.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_822.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/40000/FE92E146-0410-4246-B384-1EF9A1E9D964.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_1483.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/40000/042E4F4F-EFBE-A44F-86B6-2EA34CE7F1EF.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_1482.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/40000/458E6270-9526-9549-8F10-76DF07CC0191.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_1481.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/40000/EA3B86FD-9F33-694F-9EC1-3FA4E1FC5737.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_1480.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/40000/73D274DF-A717-3B43-A499-71AC0F5E8294.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_1487.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/40000/DC4410C3-4544-114F-A9FB-8CC1EA5ED794.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_1486.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/40000/BC294845-D8F0-5449-BEDF-A8A3BA97CC3B.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_1485.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/40000/6B735EE5-8B0A-B94E-ADD2-82E20154E8A5.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_1484.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/40000/5BD0B38D-EBAA-B143-BD40-FA59C44CE005.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_1489.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/40000/304BD8E4-FC95-7E46-8FC7-637F3AF3132A.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_1488.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260000/14050C7D-5B99-4C48-A5C0-761247E28F6B.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_797.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/280001/550FF1F8-3FB6-C14A-90ED-86E18DC1ACA2.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_2411.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260001/9E5F08F2-E7E1-2E4B-9D62-D3E076D89190.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_795.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260001/BA311E71-102E-7F48-A596-236DD771A966.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_794.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260001/5BE0E582-58F8-2F4D-9E37-5E81416C7230.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_793.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260001/5EAF81BF-D11E-3140-90D4-CDAC5E0AF716.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_792.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260001/E9A5714F-7E8E-724D-99F6-16AEA908A5B4.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_791.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260000/1ADF1671-615C-FC4F-8B75-46A669F3EDD6.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_929.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/50000/35D70AAC-A378-B041-A5AC-3D1AB285B120.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_1719.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/40000/CB8EC291-3D75-1D4D-A4EA-5FA3FC2B149D.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_1718.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/280001/4EE0A375-D303-F946-80E2-917B5E8A974A.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_2413.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260000/4F8B0C74-3CE6-AB45-BA53-96628C1BB603.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_799.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260001/C82C9685-1F72-3E46-9FA9-01A676C31C79.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_798.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260000/79C398BD-3715-4D44-8175-836E95492C52.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_612.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/270000/53FD4667-8510-AA45-A2D5-D82106094542.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_2144.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260001/14D716E4-8F51-7540-A119-151CB90EE2CB.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_613.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/280001/CD225924-4BBA-B141-A15B-C7790BD2BA3A.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_2415.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260001/99A112F4-0891-D54E-B4FD-B49C149218C2.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_610.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/280000/92702C39-D7A8-0546-B579-2CC6D348156D.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_2414.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260000/5149FA72-2F58-C640-B535-CE3D41E355D8.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_1139.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260000/8F3B982C-9F3F-8A43-B601-0C8667ADEE8F.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_1138.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/270000/7271BC19-A900-EB41-9572-4A769F623D23.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_2141.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260000/924F27F6-FEF6-3240-84AB-51305E0165BE.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_1133.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260000/4EDFF51F-9FAF-F34D-B58B-2603A72AA174.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_1132.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260000/CC6DD13D-FBCE-DD41-B268-F91FA876CBB3.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_1131.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260000/B6661F93-3B29-114E-B8A9-F9E72A8FA545.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_1130.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260000/AF7C387A-3225-434A-A707-BCEFF40A939C.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_1137.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260000/8029C5FC-7143-E040-9398-84E7FDAD6280.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_1136.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260000/094109BF-B509-2E41-8962-A828AD2956B1.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_1135.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260000/0CC3059D-4737-B349-8C93-F8330263F87E.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_1134.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/280001/49845921-EB84-8941-87F8-16735E896F14.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_2419.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260001/E0892B07-8EC8-094C-BB7C-B7AD85B355F0.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_614.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/280001/B5722612-1568-064E-91C7-62BB73555FC4.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_2418.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260001/DA7493F3-74D5-364B-86A7-933B926E1C43.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_615.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/280000/D7A97AFE-F25A-0A41-B5F4-F8F2163BD6EB.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_224.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260000/BA64EADF-A2EA-784E-BE0D-139B8488128F.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_519.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260000/41C12FEA-92E4-2A4B-8EC6-80FBCC2A3F66.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_518.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260000/B5114AA2-E0E1-6547-9829-DDE4021B9B9D.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_926.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260000/D18B76EF-97A9-B94C-A7DC-66FBCE0F6BEF.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_1009.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260000/39CE6166-097C-D84B-A4B8-E48C8DFFBE3C.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_1008.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260000/5D24B799-800F-8B4C-B392-7EBBD82BEA08.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_1007.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260000/E9EA2500-9C02-3546-B78A-002B854F6495.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_1006.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260000/98C314A1-3C4A-FF47-AA55-F8DD57058E30.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_1005.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260001/61297932-FD3E-6240-9B35-2B888D5727B3.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_512.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260000/EEDFE32A-CE43-524F-9948-D016853BC264.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_1003.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260000/F0DDB6BA-7B2D-2943-AA45-B2C0D1E6D610.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_1002.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260000/7A1A23DB-E307-3D4D-8F6A-B7A58FA92F7F.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_1001.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260000/B92EAD4E-2AAC-8845-A44F-A534A016EA6C.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_1000.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260000/F8E44970-DDA5-8240-82C3-50AE171A490D.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_1227.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260000/CA892080-8447-1D42-B9E4-FAA0CADD05B4.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_1226.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260000/2C895D86-93D8-3C43-A9FF-59D9A9582091.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_1225.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260001/5FBEB618-8CF9-C540-91C5-307F3338D224.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_620.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260000/C798FF16-4C09-8340-8154-10A8E3B9462D.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_1223.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260000/67003AA1-5E5E-734D-8FB3-D0787DDD22C5.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_1222.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260001/B082B555-2A1A-3546-8F18-638291F5E31C.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_625.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260000/71ED08ED-7352-D141-BA94-1AD54536D302.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_1220.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260000/3C20C187-5EB3-9740-8E5C-DD7FEB8509A4.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_629.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260001/DB6C539F-A99C-AB47-A6BD-07FF822F1F06.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_628.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260000/BDA4C26E-CF5A-BC48-99C6-068F5C39F879.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_1229.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260000/ACA30C0D-71A3-9D43-9E00-7CCBAF2D68BD.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_1228.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/280000/7BAC1AAB-99FD-C140-BDB8-8406139B61CD.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_2.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/40000/FC965425-135F-1149-A7A7-31A16A06DDFE.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_1714.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/280000/A95173BC-17B3-0442-BE42-AC59A126BBE7.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_11.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/280000/B068E514-ABE3-4E42-98FF-F5E7A0D69491.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_10.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/280000/691FFB3B-BC77-3048-B55C-96FA1F8CE129.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_13.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/280000/9EE88E68-1BA4-574C-9EB1-A90386415FB6.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_12.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/280000/8815E9FD-85EA-CB4B-9783-A21FD7634A98.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_15.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/280000/1BA52179-4851-794A-B59D-9C8DAA09B6CB.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_14.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/280000/3D32D047-AB29-A645-86FA-1D1A2C5D6315.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_17.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/280000/3084281B-291D-4E40-9AB8-EE762647BAEB.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_16.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/280000/8105EEF3-2C90-E54B-826E-BE90C588C498.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_19.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/280000/ABED39AC-209C-5148-B4FF-07D623815E6E.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_18.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260000/F150727D-63C0-B048-A91A-521082080283.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_928.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/280000/062E30BE-78EF-2146-A4D0-66F21BE9D0EA.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_201.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/280000/2C65A5EF-5038-6048-A479-FC994B3C0D86.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_200.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/280000/30F2A244-1587-864A-B602-A808217FA5A0.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_203.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/280000/8A234957-4506-0A4C-8F87-F6B0C2397CC6.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_202.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/280000/47DE6853-BDB5-8740-A4B7-C32AF8DC8727.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_205.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/280000/403EB9F9-15B6-E241-BD16-CD485A539E38.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_204.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/280000/3F583116-DE75-7649-89C2-EC478AAFA9C9.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_207.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/280000/D52A01D6-4A2A-8041-8C07-CB25B2C53B78.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_206.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/280000/7CD08BB1-91CE-254F-8118-BE1BC736AC10.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_209.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/280000/9B63760B-0710-1149-B6CA-DCBECF851331.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_208.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/270000/8812AD73-112B-E34B-8DF3-E2C509A2CACD.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_1573.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/270000/ADCB6483-F7B7-E84B-963F-068D42EB904D.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_1572.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/270000/B6453CD1-8DB6-DC48-9328-2FF6889C0205.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_1571.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/50000/B54466CA-C4F9-D846-A486-FE597616AA7F.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_2239.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/50000/766AABE3-89E2-2F42-92AD-CD6D738ED340.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_2238.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/270000/155DDFBD-BC08-0E45-A307-0571BC6A7F03.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_1570.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260000/4AE1282C-E612-3B44-BEC8-2947923376E3.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_998.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/50000/D4A77352-D2A6-0E4D-9ECB-8C6C64D31A71.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_2231.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/50000/59689B63-DBA0-FC4E-8B53-9A1BF9635FD6.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_2230.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/50000/6BE09F92-7B95-8A42-8F9D-15E40775B966.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_2233.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/50000/11A9F0EF-7760-D342-96B9-13E9CCF95339.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_2232.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/50000/38F059E4-0B17-A24E-9C8B-DB45022DECD5.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_2235.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/50000/EAE2AB03-87B7-BE48-9FB4-F70BA2147F4B.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_2234.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/50000/75D34687-64DB-7640-A787-39FA45DF1BFE.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_2237.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/50000/7C1143DC-D56F-174E-9E34-4C892540F71F.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_2236.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/270000/D45E5CC1-26D7-9646-8EC4-B5F283DE4FD9.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_1575.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/270000/DFFD6CBC-8640-8244-9BC2-8DF41264CE54.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_1574.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/280000/805C8E9A-B215-7540-AD5F-EE62C6D1FD5C.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_229.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/280000/108F651E-0951-9949-8990-01B9816BE83B.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_228.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/50000/12438FFC-E0A3-3946-899F-6BF9CF8EB3D1.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_2095.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/50000/9218FE6F-BC3D-9449-97E9-6BA5D170E326.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_2094.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/50000/51D7149F-6F63-264E-A31B-47F8EBB7C046.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_2097.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/50000/1712B21A-50ED-B343-91A8-05A22D1A0BA6.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_2096.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/50000/002D82AA-DE51-AD4A-A38E-E8848E199037.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_2091.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/50000/D7894828-29FA-CF47-A311-532901623AAC.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_2090.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/50000/52B42EF9-2589-BB49-91A4-E1DB96F95D14.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_2093.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/50000/0EC05892-4403-8045-86A5-698571328C1B.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_2092.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/270000/BA9DF95B-8ED2-D14A-BCE6-9D9ABFC6F905.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_2099.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/270000/8F74E344-74E5-1143-A5D2-EEB5D4B4AED3.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_2098.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/40000/D1F12E34-5622-CC4D-9A72-6A1D12ECBEDB.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_2525.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/40000/74856D27-5856-4D46-BC16-B6F9DAF74FA6.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_2524.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/40000/AB374F85-CF57-AB4A-B66B-C29D17DB5BE9.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_2527.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/40000/643A6E44-F5B6-C34F-9741-E7E25A204059.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_2526.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/40000/7969F23F-10B6-5D48-86B1-101511DEA0D1.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_2521.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/40000/EDADD804-51C6-F740-88C3-355F16B6FA75.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_2520.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/40000/1B5A25FD-99D8-E74C-80B8-51161B4E6E59.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_2523.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/40000/94213878-277C-454A-B6E6-27F8BC9B4C4A.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_2522.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/40000/7EEC1D2F-4652-1742-8986-19F8654D1653.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_2529.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/40000/587D62A0-8A0C-D54C-ABF3-D2F518D4E93D.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_2528.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/40000/F1821C67-E796-C04D-943F-9038BED51C3C.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_2709.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/40000/314B886F-AD08-4F4E-9D6E-7C217E53A87C.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_2708.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/40000/AB4EAF47-5273-D443-B2D3-7D46172426BB.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_2705.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/40000/810584FF-D168-BF49-95BE-D1FAA993D711.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_2704.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/40000/0B9EE4C6-3CB5-5B4A-ACC7-9A09D1E3FA79.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_2707.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/40000/66C1867B-C993-5542-8675-BDB940376C18.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_2706.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/40000/C5820F32-0010-3540-843A-8F5ADCE1BCD7.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_2701.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/40000/363EE185-E923-0049-83A5-BEC00003A6EC.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_2700.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/40000/BF543FC0-B720-7148-9065-3205C58063E8.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_2703.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/40000/A7BCC65E-C24D-C04A-A834-40E3B70C3FAF.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_2702.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/50000/980F30E1-6606-BD46-B6FF-4554A32905F9.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_1829.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/50000/59AA2918-3F8C-5047-942E-FEEC5860FDDC.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_1828.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/50000/F793F629-BEEA-564E-B7D1-F3D232A54A46.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_1825.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/50000/3DB2DEB0-73A4-0D47-98A4-118F28211A3A.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_1824.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/50000/681F34DF-67CE-F34D-9B83-D94ADA8E72A7.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_1827.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/50000/C387B8D8-1BA4-4E4F-8C80-BF2C3719BF79.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_1826.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/50000/EB025617-A215-6645-AD0D-00DFAC8E5AC1.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_1821.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/50000/50435B80-DCDD-0C4B-811E-E1B2F532FB2B.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_1820.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/50000/744B4D6D-9CE8-0646-9D4B-6D4845A7E3EE.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_1823.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/50000/0E52170A-2453-6443-BE21-8ECB9C61D729.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_1822.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/270000/499745A8-0226-994A-AAD3-A61E2C4AD34B.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_2147.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/270000/6EFCE1BA-D56D-5944-86FB-0E58DD74900A.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_2146.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/270000/79C0055E-E032-4F4A-A3D3-2353867CA300.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_2145.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/280001/360AB2BD-6342-3C48-B701-F148268AADE9.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_2412.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/270000/2F77AFDB-D43D-2147-988C-F9F55FBE378D.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_2143.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/270000/201FBA74-5D05-8944-97A6-86AB8CE50072.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_2142.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/280001/568097F3-6D46-A445-B68B-84430AADE899.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_2417.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/270000/117F2A17-9D04-BA48-A872-2C4DEC5C6CAF.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_2140.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260000/46A9583D-2721-2347-B599-1F2915167EC6.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_920.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260000/DB4DEF9E-7BF1-D245-B25F-FD9A42275A97.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_921.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260000/0A75B49C-CA2D-B249-8226-0EBE36E6B7F9.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_922.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260000/6A57EA4B-D2C2-4144-8AE8-92253DDBA935.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_923.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260000/E8DCF3C6-EE9B-2848-A651-A77B30F8656D.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_924.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260000/97076BAB-15FD-8645-AA09-220C96DDE0CC.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_925.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/270000/02EA2422-853F-E64C-B6C1-AC202E204EFB.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_2149.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/270000/FFF90C28-360A-2142-A937-300304300D4E.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_2148.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/50000/78724953-A717-C643-A008-1F403B1A10F3.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_1920.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/50000/F64A57FA-B24C-7746-9F08-C0AB99627314.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_1921.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/50000/BDBCFC8E-8CDA-B04F-AC01-D277C97C1710.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_1922.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/50000/F139F937-E5CC-F343-8BC6-6F8D319CEEAD.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_1923.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/50000/AEFDE6D2-C8ED-FB47-B04A-BF2DF541F973.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_1924.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/50000/DF95A68D-24DC-2E4D-9787-F6813066D8B4.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_1925.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/50000/526520E0-2959-B34E-B99D-8E87EDB03D49.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_1926.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/50000/9336D8F1-3341-4842-BF6B-D7042D9B5C37.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_1927.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/50000/0F418142-31C1-2447-99A9-0A5AAC159983.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_1928.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/50000/2818303E-83A6-A243-90D0-D78F1782D64C.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_1929.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260000/0D2A5724-E01E-CC4B-A76F-1152B4AE56D9.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_832.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260000/59BEB7E9-2C2F-5346-9805-E5AD17C7094C.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_833.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260001/BBA119F2-07AF-A541-A3A0-D433C988A0E6.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_830.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260000/E582F9F5-89D6-BD48-8745-64DB6A93FD11.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_831.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/270000/6F3962E6-121D-CD46-A4A8-750E1E771173.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_836.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/270000/CE098D9A-D996-924A-AED4-29AB3EBE5783.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_837.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260001/EB098E27-0614-1D44-9A03-F586C4E46C08.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_834.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260001/4F47F849-C076-3C47-BC69-9F756262363B.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_835.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/270000/F7DD8809-0F55-3C49-B671-19FDE08480B3.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_838.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/270000/1BCFCC1A-97C9-E245-B290-6577B208AA1A.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_839.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/280000/29D5BA05-429F-0647-A904-31F4BC921D66.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_3.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260001/47479D2D-7346-9C4E-BC1E-97C49C5A4A1E.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_784.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260001/D0375786-E3A2-5647-BBAC-C1DEC4B0D002.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_785.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260001/5103D9E4-6BE1-434C-9D51-B45972B1E1F6.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_786.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260001/A0973335-E6E0-0246-9964-AC0F69454CF0.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_787.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260001/07605FF1-6736-0A45-A2DD-BAA322490E01.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_780.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260000/6D4B2BDF-CAB6-794B-9BF6-9A46C3468B88.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_781.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260000/6B80DAD1-4014-6D40-A3D1-2624FE7682F4.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_782.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260000/938A02A1-D44D-624E-A12B-E8EED1F339AE.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_783.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/50000/3BD3B6EF-28A5-004D-A7BD-D63DED4A0C2D.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_1726.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/50000/2E322B75-E9B4-AC4C-AD5F-7E70FD07F76B.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_1727.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/50000/24791A06-0B5C-9249-9244-9CE175CB88E1.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_1724.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/50000/3B5D26A4-BB4F-7A40-831D-D5380393C8A0.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_1725.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260001/B8940124-C728-C541-A44A-3BB1B0CABD24.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_788.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260000/5F2F236B-DB4D-6A49-879F-3CCC76BB1419.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_789.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/50000/896958C0-79E8-9D45-B848-115A21E06DA0.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_1720.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/50000/C0A569FA-2C9E-0D45-ABE0-0F7FF0844E4A.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_1721.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/280000/752AFF04-651F-B341-82AC-14DE730D59FE.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_468.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/280000/66396418-8778-BB4F-99D3-920368B956C9.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_153.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/280000/DAB6AB24-F5E4-AF4E-97CB-CFDA37EFBC80.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_60.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/280000/EC068750-04F1-8140-8308-86EFAAFCFEBF.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_61.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/280000/CA565792-A77D-5E47-96C4-17E64674E877.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_62.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/280001/56302663-2377-5040-8754-B8825BD617D4.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_2438.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/280000/926FFF23-7F4C-3247-A15E-F8DBE3B13BDA.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_64.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/280000/5CD2EB0B-F767-A24D-BF6E-CE12C6E10AD5.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_65.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/280000/6425F091-A08A-7B4A-8049-115A8FC8E802.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_66.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/280000/1AA41024-8CDD-E144-BB2A-0BCAC595432A.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_67.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/280000/6E3F0FE1-726B-4F40-B577-125170BF2273.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_68.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/280000/0A88E7F5-5152-3646-AEE6-56501226033C.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_69.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/40000/963F5CFC-6925-AA46-8CCD-450484256ACA.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_1371.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/40000/51AEF107-EE5E-BD40-A5C0-CAFB172A47AF.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_1588.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/40000/2BFC9A6F-B9F3-0D4A-A607-42E3720F02B9.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_1589.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/280000/D78E96E4-A702-5F4E-9A81-F2D01F49DD00.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_406.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/40000/9B3CFA76-7239-7240-92DD-C60282ECA622.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_1582.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/40000/2DA48CA5-EB5E-C14B-8C58-C90143A10822.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_1583.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/40000/600C0377-D441-5F47-ACF6-FE99A0474D05.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_1580.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/40000/FB89E2CF-AA63-6646-B836-F4D43807AA19.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_1581.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/40000/72F7A941-06B1-A14F-ABF3-1E981D31DD71.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_1586.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/40000/C3946973-6F18-9B4D-A2AC-E2F3C2161391.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_1373.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/40000/70B77DB5-EED4-F242-A63A-F5549E38DB37.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_1584.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/40000/12D91AA1-2867-8F46-8F53-D72B107A1B00.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_1585.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/280000/8B4FB8D5-0EF7-4D4F-B2A7-A7349C1109DB.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_159.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260000/4C3C6F18-21E0-5449-AC17-6FB5D906302F.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_1038.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/40000/3594AC5C-0EBE-5148-ABF1-4B607EBE01E9.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_1372.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260000/1BE9E35E-FDBC-B64F-8C47-3551BC58CF81.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_508.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260000/972DE186-CD59-FC4C-9983-1189967F0E72.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_509.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260000/64F93C14-4CF7-8D42-A6A2-AF3029E095F3.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_1032.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260000/0A0F0808-5DA0-474F-928A-30D916D9AB1B.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_507.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260000/FBE9378A-8E8F-9345-9ED8-9998AAF3A4CF.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_504.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260000/FFC3458F-0A15-6B4C-B25E-CAAECA6D2002.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_1031.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260001/074C3A32-0843-C94F-993E-D1D2894A9D82.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_502.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260000/8F1158C5-5215-B14C-B6B6-BBBAD461A698.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_503.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260000/0E8D7DF5-8B59-8940-9B66-3F589866AEF8.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_1034.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260000/D5B8130D-571B-9C43-9839-A664CCA4655E.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_1035.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260000/C0F68AFE-B5E2-9B45-B7F9-03070DEDE87A.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_1212.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260000/44A59B6F-CBF0-2243-BC47-9C36A2279846.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_1213.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260000/E4930E3F-8112-974D-915A-5CE91BED6825.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_632.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260000/D107A8F2-269A-C74B-A0F8-7B50300221EA.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_1211.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260000/4EBD53D4-A570-934E-91D0-304ADF283812.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_1216.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260001/CC0BD1F0-D0C3-C54B-B767-9DBC33E2DFD5.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_635.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260001/F3CD3CF8-01E7-7E48-B401-45BBA32ABD24.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_636.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260001/5BA90227-AC11-9B49-BBCF-68673552BDCA.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_637.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260001/F600A685-1E97-B74B-B036-E78F60649BB2.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_638.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260001/3FA892C7-A34E-FF4E-BD05-F768D9875E65.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_639.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260000/3A0B647F-6C7E-F14F-B7A4-D868D0A26900.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_1218.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260000/63BBE8BE-43F1-B147-947E-D90502EAA248.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_1219.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/280001/736A2643-3C20-214D-B88F-95436AFBC31E.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_2416.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/50000/F582502F-E97D-7E46-A059-35232FD0951B.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_1728.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/50000/AB5AF82C-CE5F-2046-AE2A-B85583C7FA64.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_1729.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260000/E4DEAA82-9B5D-2246-B427-DB5E9F8EF91C.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_1106.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260000/5D00E6FB-7779-5340-8AA9-0843E46E56E4.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_1107.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260000/D0E88EE8-814D-0247-82CD-606BF11977FF.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_1104.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260000/5D7F6873-F43D-6F47-BF87-794F0A9CE54F.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_1105.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260000/EABF8AD0-4266-EF4C-8BB4-BFDD11F9B68A.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_1102.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260000/B630BB21-A444-8B4E-BF3F-BC44A76896A0.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_1103.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260000/4B6478CC-A4F4-8245-B131-A1690BC3A13B.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_1100.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260000/7F094F18-491A-714B-9B8A-F2D007903268.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_1101.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/270000/772CEC4C-4C41-A447-BEAC-B96A6795A5DB.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_1458.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/270000/0FAE393A-1724-4040-8E18-17E700961A98.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_1459.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260000/5D68D3B5-26CD-D54D-9909-424E597397F8.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_1108.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260000/A02D8765-0B73-4A40-AEBE-2A966BCA0833.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_1109.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/50000/48D439B5-DFF4-5C4A-8CD6-E360D698F87F.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_1722.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/50000/00F7FBDC-4F79-7C41-A177-30D4861B666B.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_1723.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/40000/5A8C9876-8D8A-7949-9418-1913EE5F795D.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_1577.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/270000/C3D4207C-5C7B-BD41-B936-662945ADE3A3.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_1576.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260000/52E2DE07-CE68-2F4E-A498-6F83F6D9A325.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_959.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260000/FD96430C-89C3-1F46-B046-8FDC8DD77782.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_958.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/280000/05112C7F-FC9A-5C49-B7A0-7A93C394AA0F.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_461.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/280000/02CB85A0-C754-3043-86D8-79E8F867D63E.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_216.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/280000/BB83DA70-E7C6-874B-9449-C499584C8638.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_217.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/280000/C7BB570A-B6C7-0C4A-A197-716E50B641E5.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_214.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/280000/BBE72D43-20B5-D34B-B3BF-46ABDEBF37D4.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_215.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/280000/3CB7EBC2-0BB2-F149-BDDA-255BD6C5EF4A.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_212.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/280000/237F02D4-388C-4342-83B6-034C9CAD72EF.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_213.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/280000/54596320-ABDC-264D-97E3-9F674BC69539.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_210.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/280000/910EC50B-4EA8-804E-94EA-1624244FDEC4.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_211.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/280000/9A185DA7-20BB-2C4A-BD2F-14EC9D9F14B1.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_218.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/280000/72821C09-2508-EA4C-ADEF-A71595A35DB0.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_219.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260000/D238B534-34EF-334A-B204-7D7AEE3A80AC.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_957.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260000/8E85BBC5-3D17-7F46-B4D0-39EBCC8DC2CD.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_956.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/280000/61072E6F-5281-9D46-B4F5-11A84023D056.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_462.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/280000/33C5C77A-FA35-0140-B1E5-E2785EDBBC68.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_4.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/40000/CD61B939-25D3-154C-92F3-6201070FA81D.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_2550.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/40000/DD3FF9FE-9E31-4B4C-8728-FA7FC62585C2.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_2551.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/40000/7B5A9CEF-49B9-3C44-9854-53C30DCBC9B8.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_2552.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/40000/0F39BF5F-6479-2341-A3C1-0D8E95AE76AA.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_2553.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/40000/0FA05848-32A6-D745-BF04-2DDD53DA70EF.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_2554.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/40000/EF77FAAA-3383-0947-9E47-0E41B3764F38.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_2555.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/40000/31843EDA-4AA8-8A46-996A-9870006FD5F8.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_2556.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/40000/956F0E36-B040-0A41-BD19-290D4BC554DA.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_2557.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/40000/78FF9790-FD7B-2C4B-A739-EFAD1EF29F16.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_2558.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/40000/24001BE0-2A8C-DD49-B7DE-96F6AA3CA97E.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_2559.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/280001/456B1D77-FCD0-E74B-B0E0-C7C9A99D2E10.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_2396.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/280001/A53B88C4-F510-984F-93EE-7467794BC927.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_2397.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/280001/F23E8EF8-6DE6-424B-9956-62E6BC539A87.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_2394.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/280001/71AB8BFF-95BC-9E4A-818E-9BB194C232DE.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_2395.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/280001/12DA204F-727E-B04D-9AC3-D128F046B249.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_2392.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/280001/5D5A1CB6-CCF8-A34C-BA46-755EE4E91735.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_2393.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/280001/DB9822F6-7B60-E240-85D3-99875E39C41C.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_2390.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/280001/2A6BDDFA-61EB-AB4A-BCDF-C29CDBAFC770.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_2391.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/280001/7E826E89-C71F-C84B-8DA0-45199BC8D751.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_2398.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/280001/4AEFAB28-329E-034D-89EE-73959AD7BA25.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_2399.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/50000/8D4B7DEC-E5D9-3947-B93B-1CA91AB61FEA.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_1858.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/50000/25227781-B619-0040-8B63-7AF71A25C778.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_1859.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/50000/A6F16CDF-42FF-C647-9C2C-91A1C415C1A7.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_1850.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/50000/E44DD388-617D-E44D-81C4-428F31163DD6.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_1851.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/50000/33A8896A-A86E-4948-BA27-754FFAE2636A.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_1852.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/50000/C9A5D6E6-D44F-B74E-81E2-1285B57667B4.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_1853.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/50000/2BCAF976-9387-CD40-A0B8-FFF97937A963.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_1854.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/50000/5912AB28-A5C5-624E-9F08-CF319BC3F913.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_1855.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/50000/3DA1DE71-512B-0F40-B449-9EDEEA71400D.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_1856.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/50000/644CA064-521C-4544-A7DC-54E06E320AAC.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_1857.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/270000/D02FDCA9-83F1-E045-BC16-264BE840BD78.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_2172.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/270000/BCFA79A8-6122-8440-90AE-156555A08DD7.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_2173.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/270000/F982BC7F-1E23-C64D-8350-827C37512842.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_2170.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/270000/3141C0EC-56DC-C540-892D-A1F294658422.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_2171.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260000/04939F8B-6606-3F4C-92BF-6D7D950376DC.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_919.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260000/6748CB05-B1A6-4B48-8A13-97E42C7818AC.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_918.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/270000/4D20D847-60B9-9744-9B3D-6232D51C1080.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_2174.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/270000/F8FB2CF5-D282-794C-BD14-267E386B989E.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_2175.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260000/33BD1EA4-E9BC-9B43-B35F-13A9F54341F0.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_915.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260000/D00F20E0-C2D6-0B45-8E13-9F2B36D4AAC7.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_914.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260000/617174AF-3E3D-4048-B2B4-26A7E913686A.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_917.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260000/CD2293BB-444F-854A-9676-DCC199C97B77.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_916.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260000/B836B7B6-DD1F-4343-849D-F4981A9DE23C.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_911.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260000/C4B4512A-822D-4641-B1AA-97465A2C4AE3.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_910.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260000/C37C2689-BC66-0142-A6B4-6E6E85635F25.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_913.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260000/EE07C640-13D8-504D-ADD6-23248EB15791.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_912.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/40000/DD6DEE63-B475-8946-A1C7-0E8504E2D24D.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_2604.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/40000/3C34AB53-69F2-5A45-90F6-ABEABD699B96.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_2605.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/40000/B5D3D7CF-D58B-8E43-A8F5-81A8D1ACA449.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_2606.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/40000/188304D4-B690-A642-9655-3E2F838DCA48.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_2607.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/40000/6F7394B3-7D46-F047-83D6-FEFD51544C70.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_2600.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/40000/A6579F5D-994C-F049-B5C0-6A766A4A3A57.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_2601.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/40000/A89D3D36-0256-B140-8FB4-E520787D1072.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_2602.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/40000/0A1B4C37-EF54-F94B-852B-C0F175C11FE3.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_2603.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/40000/EE216A7B-035D-8443-B402-07885587D353.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_2608.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/40000/E32BD293-F118-6D42-B381-5F68D9B6BBA6.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_2609.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/50000/F30D27F1-EE78-A24D-829D-35F0BFB92E9A.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_1933.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/50000/039A0D62-9B2A-8742-8B00-7AC47FA624A2.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_1932.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/50000/33171F63-A5C2-9446-B275-C9961A27F86B.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_1931.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/50000/47120EB8-1CED-5E4F-B81F-F0EDCE877BF9.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_1930.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/50000/D70729F5-EEAF-7A4C-BB3A-3B87E69EBDEE.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_1937.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/50000/9156D725-5CFB-2D43-A0CB-B34D595B93EF.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_1936.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/50000/632AF3DF-E327-D441-9764-F48C1707BC54.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_1935.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/50000/A49E0765-29AE-7C49-A606-6A31E3104749.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_1934.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/280001/6F660949-9D3D-E940-AE4F-7C500A4EC2CF.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_1939.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/50000/83A5586F-CCD4-314F-9BD0-C542A7EADF41.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_1938.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/40000/96C20EC4-160E-7044-8F6B-D7636812B041.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_847.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/40000/1485FFDD-0485-F746-811C-8E21E3C1BBF3.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_846.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/40000/02C75949-0A28-C043-9880-79C26C44C16E.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_845.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/40000/AEC7C67D-0B3F-E240-9BF3-0A58E4D98D95.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_844.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/40000/74278BB0-8215-4B41-9E9B-92FAC183E948.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_843.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/40000/CF686A6E-FFCF-A14C-B6C4-C26FB071A31F.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_842.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/270000/DDE4B256-D48D-FE46-BB03-06A66562BF33.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_841.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/270000/645ECDAC-A82C-6A44-89BB-C33E3CBDDA16.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_840.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260000/344A37A7-B8D1-6D45-996A-A2CE8527FCC6.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_1426.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/40000/A635A497-23F3-EE47-B5C3-43EEBE4C1B76.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_849.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/40000/6BAD02AE-2055-044B-B251-926B21A3ACE3.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_848.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/280000/694F62C9-6FE8-B044-97EA-29354AB4ACAA.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_459.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/40000/27A1BF16-3135-7D43-8EF2-B317786944D7.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_1587.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/50000/F8FCF14A-F982-894F-82A8-8F9120023035.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_1739.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/50000/F0BCFBC8-C64D-6745-9772-57C26C3DAE7B.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_1738.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/50000/C2BF4708-F26D-3A45-9AFB-A66DE6A3D3A2.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_1731.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/50000/B7E9B35E-29A2-2E4C-A1D2-751B7A40D740.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_1730.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/50000/8B93806A-254F-904D-8C92-517CEB85604A.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_1733.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/50000/36A5297D-A57C-9D48-A01C-18A924989F47.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_1732.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/50000/0545BFCD-5BEF-6D40-ACF7-BE603C3F12D6.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_1735.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/50000/C8A13D70-09B4-B449-AAF2-E469E5533FEF.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_1734.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/50000/87D71E38-F1D4-A744-8350-03A7D9B97F09.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_1737.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/50000/4378AF0D-03A4-CB40-AFCC-516F0562E52D.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_1736.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260000/2D7E42BD-9009-1B46-980C-FFA2C576E038.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_1039.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260001/BBCAFF27-E4C1-7947-9655-AA5B82124077.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_753.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260001/BD23073F-DDA6-EA4B-83CD-B3BC263B0CCD.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_752.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260001/0132510C-7C63-5348-AA7B-921E8ADDE68A.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_751.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260001/9E80ED55-24F3-CA41-9532-9B749F9B4098.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_750.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260001/27AD4352-6DAB-9141-92A2-0D147AB0E9F3.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_757.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260001/F22C3EE0-53E6-DD4E-82E7-246C0026C581.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_756.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260000/FC6611FA-6FEA-684B-A265-A107F66C5FBC.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_755.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260001/100B7D5A-8C8A-D040-A096-1C0FF204C97D.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_754.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260001/99C87A34-D1DD-6C4B-96F4-E09AC68BD220.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_759.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260001/ABBE6348-21C6-154B-9E10-B5FB86F1B967.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_758.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/40000/62A4A9B8-957D-754A-8DF1-0F71FF7A1EC6.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_1595.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/280001/1544859C-5FA4-AF49-9CCA-855342A503E8.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_2405.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/40000/5253C330-5A8B-BA4E-B216-43433460561B.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_1597.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/40000/718A5D84-796C-C34A-9C64-0F0A1FB7B1E7.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_1596.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/40000/5153D42B-8D00-184D-B523-42E21B54D820.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_1591.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/40000/56B174DC-E675-524B-B852-6B9399286182.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_1590.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/40000/A57A41E3-7963-D440-A469-BD8A57D5C684.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_1593.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260000/E73A9833-0D85-7B49-B2D2-890451650962.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_1033.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/40000/08745D79-6A23-AF42-8433-6008A776AD95.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_1599.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260000/58A3CCC3-DE4D-8449-8A56-92C7A31E9961.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_1030.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260000/054042A4-058C-194E-9756-038004285B82.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_1025.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260000/43ACE3C1-3FC8-4447-B417-9D503921651B.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_1024.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260000/0FA08DDF-4738-A64A-A1DC-F50BFC85EAF2.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_1027.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260000/9588F1C9-0DBA-C441-9A95-480200CA5AC5.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_505.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260000/4FD1DC0D-0625-0949-B70E-73321F371416.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_1021.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260000/DA1E1E19-1331-494A-8B1D-83C091A94EAA.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_1020.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260000/DAA57D89-52A1-854B-B002-B26BD4727C87.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_1023.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260000/9DFFF2C0-7F28-B94D-BA36-554216BAE7E6.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_1022.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260000/A13518EE-FD80-5044-91B7-797E6AAC58C5.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_1036.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260000/C7FDF19B-7FAB-7E41-8A78-E351FE297DB8.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_1029.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260000/4B9368B8-7B67-8249-95BE-4063EFBA485E.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_1028.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260000/FF44DCF8-BBD8-6445-961F-D5A6F554161C.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_1037.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260001/A8C85B1C-E67C-8F4F-9D9B-DC1120568F96.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_500.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260001/BB94203E-618D-804E-BD7D-8B7FE8246001.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_501.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260001/421D000D-E11A-EF41-ABBA-D60501794BDF.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_605.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260001/CC48822A-FAB2-0A4F-8053-43ED94DA9F98.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_604.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260001/AFDFCF39-A808-1644-B451-2E508DF1565B.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_607.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260001/604489BA-51EB-9444-BEA1-9A09EB674A4F.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_606.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260001/DA7357FB-0E79-FF4F-A71C-269A58E1569E.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_601.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260000/AA64F164-1C97-ED4B-B5EE-18BDE8F329D2.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_600.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260001/D5F7CE8D-5A24-BE43-874B-A5D58344FE38.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_603.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260001/101A6F79-0577-394F-8708-8FA7DDF41870.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_602.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/270000/76637C0F-DA9B-9846-9191-CF2D69992289.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_1205.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/270000/0A6B439F-D8A7-4A47-AC17-2F5C83B91E41.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_1204.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/270000/FBAEED3B-CC99-D54C-92EE-A244E37DBCAC.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_1207.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/270000/C3A4ADB2-15D7-0042-B46F-D123F3D54E8B.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_1206.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260001/159044B9-639E-194F-AB88-92D2A30D743C.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_609.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/270000/8EFEDAF9-3692-0545-8999-743DC472C7E1.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_1200.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/270000/5A9594B5-94A6-4446-9AA2-61BE6E5E8F73.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_1203.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/270000/7893F94A-155C-904F-95F2-5D09E7168FDD.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_1202.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260001/8CF0C780-D255-4F43-8ABF-5E32FB393B39.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_633.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260001/424FAD6F-027B-8B46-938E-EE507DB2F632.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_634.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260000/CAA2071E-9EE9-064D-9EF6-A76DF66636A4.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_1217.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260000/5428EDF9-5D53-1347-AB3D-28B63CA3C553.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_1214.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260000/E7431576-0AD1-CE4F-923A-9193BD2AAE63.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_1215.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260000/CD80BD4E-E686-0D40-8BA0-288E7A8A8230.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_1111.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260000/5707D683-DABA-5841-AC0D-123418504C5F.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_1110.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260000/AF56C18A-3953-B94A-BEA1-20A89310D185.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_1113.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260000/74B0E398-11C5-3E43-82D9-D8135B029FB7.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_1112.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260000/400F5DE3-2559-8944-BEF4-669F878B5709.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_1115.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260000/A28FF085-255D-554C-8D2B-F1096C67F4AF.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_1114.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260000/0D0AF398-45F9-4C47-B639-DE12528362BC.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_1117.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/270000/9D1AA813-92F1-ED49-BBB7-0C783DB82D02.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_1440.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260000/0A529A9C-C6DC-B04E-BC81-A64144461C4F.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_1119.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260000/3D98F7B7-D150-0E4D-ABFB-ADFFA48838E4.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_1118.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/270000/C99A8FD6-5171-914D-8846-4C4F99F154CE.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_1351.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/270000/32184B99-EE5F-5042-81B4-B6E616F2CBF7.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_1449.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/270000/C4D2E2A5-D720-F047-B379-0C95D0F5B9AC.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_1448.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/270000/FF4E9ACE-7C6F-0147-956F-F9BE66AD653B.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_1350.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/280001/6A7003C1-0C2A-CC4B-9A88-F185D7F189C1.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_2401.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/270000/23266930-C608-104B-BB3B-64C66A052D1E.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_1356.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/280000/4B28CBC4-3AD8-C148-AFFC-F5DE25171F6A.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_463.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/280000/54642F35-2F1D-5246-A88A-5AEB953C4104.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_489.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/280000/5139A841-C8CB-2846-894F-DA1E2B8975BA.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_488.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/280000/EECEB2CD-A479-C841-AAC1-727D8718E85A.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_487.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/280000/0C516FE6-FF9B-DA4E-A1AB-957CFB4D6594.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_486.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/280000/F4A39519-1486-9644-933A-C57AB565148A.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_485.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/270000/1E63B22C-15E1-C643-BA6B-C82ABE69C07B.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_1354.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/280000/8409731A-6A71-DB4F-9E2C-0A1F3AC29507.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_483.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/280000/8C76ABBA-AAB0-534D-B170-DFFCA3D8CC96.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_482.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/280000/6E4BF0C6-F8FC-F247-B200-667061C7BDC9.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_481.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/280000/0AD051F2-F0E5-F043-BC87-CFF20C50F3E7.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_480.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260000/878B8FEE-D5D7-1842-8D54-3BD272EC410B.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_991.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260000/9EEAA6C4-CF38-B444-B06E-D492452136FB.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_990.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260000/0BA377C2-8CA6-254D-889B-789E6AF31377.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_992.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/280000/659B6F02-CF4E-D543-88B2-07729A518490.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_199.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/280000/56658207-CB33-9B48-8255-204D395CC8B8.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_198.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/280000/75636CB6-DAB7-B241-91BD-C373C8C98215.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_195.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/280000/29DEE188-838E-6D47-B823-EB232807D25A.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_194.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/280000/E71144A2-45F4-224A-AC67-1E25B4CFD851.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_197.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/280000/CC775F34-D231-D645-852B-DA5114CF45DE.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_196.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/280000/69E8DD9C-8823-F74C-B072-B415DE242247.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_191.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/280000/CF691789-B470-214B-A784-F1846B20E316.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_190.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/280000/65BA3F02-B6E1-3A42-B123-9CEAAEFD4D46.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_193.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/280000/F2C2DD6D-DF78-E242-94E6-5463E645EEC2.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_192.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/40000/5C26C1C0-971B-0F45-BF43-C3F5F71CA142.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_2543.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/40000/A0AC2571-7C1D-7746-B0C2-4D1FEC83E7BB.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_2542.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/40000/102777E0-8FC9-0444-8FB4-230BBE9C4445.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_2541.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/40000/B780E093-CA8F-7F4D-BFC0-9461621A524D.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_2540.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/40000/CF96658F-2F9B-6942-AC69-39C7FDE7FB49.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_2547.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/40000/D937490B-7554-3B4A-9C34-87DD708E2F9D.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_2546.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/40000/FB88D564-C052-1447-8A8D-E9F6C582BC38.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_2545.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/40000/908155CF-D516-624E-AC43-A459F33D7A83.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_2544.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/270000/6399117C-EFCB-D249-B6C2-73C6F089DEBE.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_1274.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/40000/F34598D7-1445-E247-8D07-5E0CE45B586F.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_2549.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/40000/51D8C244-077D-3C4D-91BC-42818EE125B4.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_2548.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/270000/6FAF0656-C512-524D-9CCE-295B177F00E6.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_1456.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/280000/98A22D76-8435-DD40-B43C-51AC5EDFD67E.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_2381.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/280001/ACAC22F8-CA0F-1849-8DDE-ED31D7B572BB.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_2380.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/280001/02ACAB89-FB02-B740-9B4D-5FC8F4084F37.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_2383.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/280001/52277628-A386-E440-AB45-1232379939BE.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_2382.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/280001/3029DBBA-BFC5-5A4E-9773-22E70B503951.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_2385.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/280001/CE455ECC-F4E0-7B4C-988B-EF7275B37413.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_2384.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/280001/EA76AFCF-85E7-F642-80E3-39F045244BD1.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_2387.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/280001/996F3725-0C7B-9A49-8096-A9E013A84F5A.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_2386.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/280001/15A1E203-965B-6246-98A4-700240CBC4D8.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_2389.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/280001/BD6E88DB-B745-9142-8C75-B9332585FCB1.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_2388.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/270000/2957CE42-9239-0E42-B752-99165F1C31FB.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_1451.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/270000/453E0211-463B-A14C-93F4-CC07FA4F590B.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_1452.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/270000/12B3FAC4-411F-E247-B1B2-3F2C52491353.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_1453.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260000/96B959C5-C3C1-B54F-8F0D-361C1B5581E5.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_902.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260000/1B22EAE7-770E-7E46-B462-A6ACC9BC9550.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_903.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/50000/33B6B7E9-23E9-4A48-BADB-F8ADEA0E96BB.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_1849.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/50000/B76C6DC8-6E4E-474E-B439-76A5F6419841.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_1848.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260000/291B19C7-CBE0-F049-B8AD-A1D6F7DEC095.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_906.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260000/F4252F90-A59B-1B4A-9756-03BBC6E9511A.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_907.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260000/484D1BBF-3A31-F645-94BF-3B872B1F74DA.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_904.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260000/DCF6BCC1-BCDD-FE44-B87B-09EDFA782A89.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_905.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/50000/16EDEAC5-FBB0-A148-B222-AFF73A8E44FC.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_1843.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/50000/8B06A349-618C-4940-B152-AAC8C601E928.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_1842.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/50000/4E1F7C0C-817E-E54E-9E7A-4294D63BB4AB.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_1841.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/50000/75EBF115-632D-5D44-B1E2-3D23A268DAB2.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_1840.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/50000/3CD8B598-8DD6-8740-9FB4-3C328A61A19A.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_1847.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/50000/0C974F8E-BD1C-4A49-9D6D-557FACCABCE3.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_1846.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/50000/44BE7524-B888-344A-B16A-C1A1F0297F28.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_1845.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/50000/26DF7E0B-29B6-5D4B-91B0-AB6D4D356461.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_1844.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/270000/66601FD9-2810-0346-B083-95A39647DA09.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_2169.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/270000/A98E85F9-3486-6246-9367-28B509B2DD27.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_2168.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/270000/FDBF8811-17A5-A845-857A-0B49DCBC9F40.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_2165.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/270000/AEFC43B0-AB8F-B64B-910D-CC5B7A7DE5D3.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_2164.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/270000/FB69BE96-0612-164A-A541-0C2C80629F4F.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_2167.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/270000/6A39CACF-0B96-0746-B3AE-5F7F697A1AE7.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_2166.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/270000/10E7173E-B21E-0F4B-A5CE-A76D61DCC8A8.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_2161.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/270000/69058AAF-5238-CB4D-A5FD-1F6B42A064E1.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_2160.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/270000/A1365F7C-2E75-6042-8EF0-2A7075072DF9.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_2163.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/270000/B7700DAC-A82D-B041-9C53-D624F86279C0.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_2162.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/40000/F17D0797-0565-6D40-B132-6D83A60EDD77.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_2617.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/40000/93B3B947-562D-0E45-9E2D-B98678275D42.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_2616.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/40000/42E7E13C-7D39-BD40-A9CB-C4A2DAB4A1F9.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_2615.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/40000/E1DD5D2A-F16B-4C47-85E0-E471990FBD93.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_2614.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/40000/51B8D714-A012-964E-9BBF-CFF81C43916E.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_2613.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/40000/2DA08D9F-66C3-9646-89DD-7658A6916D7D.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_2612.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/40000/B3BB771E-CD30-9D47-BD01-50823E4ADA0C.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_2611.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/40000/061E479C-D383-BC4E-AFAA-0B352CD4C611.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_2610.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/40000/4A71564E-9B4B-C645-8925-A93E51CC686B.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_2619.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/40000/3630E6C1-DF4A-AD40-B3F5-C73B14FB1283.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_2618.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/50000/CDCC02A6-6563-4C4E-B37D-BEC9A2F2ADF5.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_1908.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/50000/F3B2CE5B-1C09-B843-BBEE-A9EEA3E3F256.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_1909.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/50000/5B58B58D-5A06-814B-8BD3-205F840B4CA0.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_1906.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/50000/FD62B845-E9B7-E748-A63A-3F71235309B1.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_1907.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/50000/B0A56E39-25EE-4E40-9E38-0CDF3FA93467.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_1904.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/50000/A85BAFD4-F0C8-614B-9AFA-A5F849993A46.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_1905.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/50000/A178B010-6BA2-F545-B7C3-499F13401B27.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_1902.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/50000/5935B0B9-79DF-C846-A117-EFA265B194CF.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_1903.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/50000/7123D790-2C43-D54A-859A-EE1D51A611E2.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_1900.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/50000/81A0994C-4107-E944-871D-362598684DE8.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_1901.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/40000/F01D54F3-D9E9-4245-A4CE-3C08AE1B36C2.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_854.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/40000/60C6A842-7DB4-A04A-A9B5-7CF3213D0103.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_855.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/40000/7D28B021-B649-6449-9AA3-15B440AB63DE.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_856.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/40000/7833DE5A-B44A-6444-818B-7F759FD7291A.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_857.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/40000/D5F63795-7556-E443-9CAA-C93E41BD211A.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_850.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/40000/0CAB152F-E6EE-EF4B-B910-2D68A62A8155.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_851.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/40000/3033961C-7346-DC40-90C1-CC63E6DFD230.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_852.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/40000/FC99B013-5F5A-E340-B6CE-CA73F00B571D.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_853.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/40000/9396E4E1-6916-4846-9528-1F0A6784C316.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_858.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/40000/C898EC20-8FC5-C044-B8F0-19C56210ACBD.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_859.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/280000/5D35B8B0-2185-0B4D-99A3-D57FCF24A852.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_6.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260000/39823DE6-67F6-9C43-AEC9-92532BB824E2.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_740.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260000/74CFF46A-8E18-F442-9EF7-FFA89A643361.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_741.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260000/353CCCE5-5553-654B-B082-C3D0FB8ABBCC.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_742.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260001/B231E1ED-2387-8B4D-B53C-41BD295A63D6.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_743.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260001/25F39757-6552-BD45-9F24-3238BCC39522.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_744.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260001/DFFA90FE-9396-CB40-B406-6F80316C23A7.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_745.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260001/A470865B-E7CB-B249-8279-E86B6CABF8A0.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_746.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260001/82F81233-1EB5-1948-B9B7-BF8CA8921A4B.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_747.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260001/912ED4B1-B1EF-4D47-9837-427D63074F9E.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_748.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260001/7C13039A-C59F-9145-B164-334DC84FB421.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_749.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260000/1761125C-EF68-A94F-846A-A18D0CDFCE3F.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_1050.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260000/EB12203C-E838-1247-A5DA-345AECB1ABEE.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_1051.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260000/5AAAF26E-75C8-AA4A-9308-62C47026F7DE.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_1052.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260000/683BCCA6-104B-0E4A-B24B-C219D4EA0AB3.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_1053.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260000/D6487E5D-E70E-514B-AB1E-24FB3FE8B085.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_1054.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260000/F025C53D-C4DE-C948-A347-AB8DA030962E.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_1055.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260000/62337F4D-650C-2141-93EF-49D2AB885E3C.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_1056.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260000/EA6D6A7E-E8E3-244F-B16E-94430270DAF6.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_1057.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260000/2A5DB637-6AE8-874D-B4FA-B2CBBB079936.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_1058.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260000/7C0ECB13-3682-8F45-8733-7F5846EC9733.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_1059.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/40000/7084E2CB-C70A-304F-A445-0F0400DAFBCC.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_1696.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/40000/2545DA78-3190-E441-A2CC-A43C6B6BB455.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_1697.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/40000/CF8186A3-FDDB-0A4A-B042-961134A86D3B.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_1694.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/40000/3B82C5AC-8F0B-E243-A80F-7E746F58D670.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_1695.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/40000/AEC51214-6F0C-6F47-A370-632A96F5F6F0.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_1692.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/40000/C6E785FB-9A0E-E34D-9ADF-A557E4D70191.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_1693.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/40000/0EC6762B-32E8-2048-BE47-C8493FD2B788.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_1690.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/40000/558C1978-5AE6-244A-BA1E-C4E229A713A5.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_1691.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260001/23A9DDD6-E1CD-5349-A94F-032C56A6E2FC.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_715.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/40000/86F45ACD-18FD-5E48-B384-FE7AE5BC189E.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_1698.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/40000/0D9AAE2B-1D04-E145-BDA2-C31916DDDDC2.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_1699.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/270000/9ACFAFBD-3203-3245-A1AA-45DD39B646BF.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_1278.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/270000/09FC6068-C56C-1241-AB1A-4D16CC13AA48.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_1279.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260000/4D82EB41-5453-A44B-A1A1-944D619F4150.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_618.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260000/EB3D557D-DC86-724E-93AC-829420FCF0FA.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_619.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/270000/5EAB855A-671C-F74D-AD28-EB2120EFDC3B.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_1270.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/270000/64445409-7F6F-1543-8F61-0D50D7A2FD5B.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_1271.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/270000/D2EAE8A7-E8FE-EC41-B27C-0326A98CB3AC.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_1272.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260000/FBF8EF29-400F-6743-A7B9-BBE995E117EC.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_611.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260001/19CA8D87-5301-BF43-8549-F6BDD757EA84.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_616.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260001/723836E2-1D7C-AE41-B29E-DA1DFB6B9A0A.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_617.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/270000/FA2272DE-D560-B24E-84B4-7F0BBE8BC476.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_1276.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/270000/90F0948A-FC93-B34B-B347-9AE57E6EEAD6.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_1277.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260001/81F6D146-9F20-BE4C-BF52-88797481E9CF.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_711.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260001/775BA9B1-BAB7-E04C-9185-944904A13E40.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_710.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/40000/C462AD35-787C-CF4C-91B1-9AF22A2ABF73.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_1491.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/40000/B65A8A9A-62E8-5846-869E-57C17E959569.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_1472.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/40000/50764BF7-36A9-4242-B4A9-70483845C0BA.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_1473.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/40000/BD7B8DFE-D955-1044-BAAE-3D8722B8DF92.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_1470.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/40000/4706A5CB-7720-0D41-B0CF-FCEBC2FAFF85.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_1471.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/40000/3F05A909-B4EB-EF4F-9B4C-773304D32C3D.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_1476.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/40000/37DBCE3A-4C38-4B49-9B2F-10850BEB466D.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_1477.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/40000/7C616322-F069-4B42-BB28-6725DEC0A8CB.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_1474.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/40000/41370913-ADBB-E640-8750-40E3AC66DB74.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_1475.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/40000/7F39DB1C-682C-1141-BDF2-33F1790FE4D7.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_1478.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/40000/3ECD6DEE-49CB-4E42-BC59-30FD6DBE6635.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_1479.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260000/94CCB01A-A964-9D47-97B7-3A2EFA877FB6.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_1304.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260000/86B11EC1-956F-7F41-9FAE-AF52D33DA58E.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_1305.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260000/FA17987A-CAA7-1443-BA53-71FD5B0343D0.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_1306.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260000/017C1808-933C-0D48-95B5-A8B9A0B16601.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_1307.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260000/F23D5197-EB98-A348-8A04-CAD6B1B85F1D.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_1300.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260000/15A8D21E-5ECA-8B49-9249-5DC98BB2D58F.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_1301.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260000/7DCB606A-9817-B840-9D46-D28E418DCF84.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_1302.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260000/EFB943AF-5BEB-2C41-808D-A5DDD3B98B8E.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_1303.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/40000/7217C822-C254-BC46-B7D2-5E81BEAF7ABF.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_1497.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260000/720ACB10-EC06-2B40-849C-B7B7ED8AA3DF.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_1308.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260000/3F1AD469-F31D-234D-89DD-3E8C0F5410AC.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_1309.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/280000/5BE369B1-7475-AB4B-9EEC-5F210EBE446C.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_498.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/280000/B4658F08-0AED-AE47-B5F0-7B4164D67F08.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_499.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/280000/7377DD05-A345-654C-8099-62F9E08660D9.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_494.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/280000/46B16685-9D70-1846-BAF2-39C85944DB92.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_495.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/280000/0E6FE254-C536-234D-A297-CB1A57F939D1.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_496.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/280000/2AC90BC4-7069-CE4B-99A6-8ADF9E6859E3.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_497.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/280000/B838AAFB-5DE5-9943-AC45-DCE9FEE2A7FB.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_490.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/280000/6882E1FF-A405-C746-B78F-7FDC1965CFF8.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_491.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/280000/1170575E-1F1E-B140-B1E3-AF493CA68D82.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_492.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/280000/716C6320-4741-2B48-93D8-38F3852C2DA7.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_493.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/280000/7584093F-7A23-544E-AD96-1FBF810814C5.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_24.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/280000/40AFF222-3193-7A4E-88FA-3D3D4B7F5EFB.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_25.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/280000/1272A9A7-D383-0547-8BA3-BAB56A92643C.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_26.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/280000/E3A5025D-459B-B342-BD24-6934D23809EC.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_27.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/280000/8E7B7BFF-01BE-1A44-8B3B-765ABFDB6463.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_20.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/280000/7697EE09-6B58-114C-8845-703FA01DE7ED.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_21.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/280000/0E8910A4-3CDB-4345-A554-B36EFA044004.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_22.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/280000/A381CB68-1115-FC4E-B329-9D2C90533DA6.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_23.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260000/47330DD1-2A03-844C-B7F5-F4FDF5776660.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_927.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/280000/2435FACE-C6CA-854E-9673-F7D6A9A38C83.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_28.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/280000/F9AB691D-263F-5349-A62A-76E301865A3D.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_29.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/40000/4340DBF7-1D42-7C4E-8ED0-46109D00C5D7.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_1703.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260000/DC7DBCFA-A219-1A4A-8F49-3DC9CA7AE647.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_1516.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/40000/5D2FE206-6950-0A49-8302-EAC2E6792CD2.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_1709.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260001/795157B3-7710-6348-94F1-467985B9185B.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_597.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/280000/4EB7D4B7-12DE-EE42-AB12-E03B510EEB82.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_7.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/270000/F4D96A5E-06C4-9D42-8075-45E3DFCA66E1.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_2288.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/270000/42800CE4-AB03-A04B-9862-CD650AA7244A.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_2289.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/270000/344343B5-0CAA-B544-A471-BA0AA92F5DC5.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_2280.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/270000/0252A516-E731-C144-888D-F6C3EE9CB8C7.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_2281.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/270000/14ADB68E-5138-6846-9F80-61730C50C02D.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_2282.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/270000/F47843B1-19B9-A84F-A4C7-A2C22211F3E6.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_2283.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/270000/B2BC7D19-C1B9-3C43-8B6C-C295665365C9.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_2284.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/270000/73493C0D-8355-274F-B3B7-458DD37D6D6A.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_2285.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/270000/1E468169-7AB5-5C41-AA3D-84BCB8764FFF.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_2286.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/270000/630C21E2-C765-9445-A2D7-BB4241A9C55B.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_2287.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/280000/D3D261EB-CF7E-3944-A383-29BE8BFF9BED.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_446.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/40000/4F0C3755-77DE-F342-8F09-E4D43228C1BE.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_2578.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/40000/D06C0568-9F54-9043-9777-7A0E781DAD30.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_2579.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/40000/EC42422A-A566-6649-83D5-44C47B4C50EA.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_2576.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/40000/FB308A34-6D19-5049-849F-011EC5F8FECF.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_2577.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/40000/7B6A9F56-0BF1-AE4E-BD49-B6FD2115CFA5.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_2574.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/40000/0BE7002F-6B4C-6043-A5F0-45402770F212.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_2575.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/40000/179B7CA9-E5F4-9748-B0CA-563F89AFF0F2.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_2572.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/40000/0A33C8A6-4074-5A46-AE60-97FBF008E27F.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_2573.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/40000/5380387F-C8FE-FA43-82E6-8C7303C3A220.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_2570.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/40000/658E7641-F1C4-E94C-96F5-67D10C6EBECA.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_2571.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/280001/C75C2795-0391-524F-9C60-322B3AF2F10D.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_2378.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/280001/DB258A13-3C08-6D40-9FF4-191FF1815FAE.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_2379.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/280001/57B34F12-7CAB-704B-8463-D562CBF3E007.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_2374.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/280001/0E1CF49A-68B9-5947-BA4F-B9DD8582EA40.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_2375.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/280001/69E33772-2BA4-B540-A82D-B9300EF52A06.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_2376.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/280001/43BC6D98-2C14-BE4B-A30C-07461FD27133.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_2377.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/280001/FBF0E4AB-E33C-2D4A-87AF-4DC8D0EAFC0F.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_2370.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/280001/671FBCF2-8A51-C44C-BB78-62C10C5F28F7.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_2371.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/280001/1C9C818C-9BC9-DB4D-BF5C-921A697BFD95.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_2372.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/280001/610B451C-7C51-7C4C-8BA5-4E0BE56B73B2.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_2373.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/40000/0E0530D9-4FFB-194F-A78B-15A758C7DAE8.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_2594.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/40000/52F98FDD-4496-374C-80F5-161B17895FB8.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_2595.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/40000/67ADCB3A-8913-C540-ADCB-1775C56DF585.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_2596.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/40000/8A6BEA0A-B059-F149-A4DC-7E775490BAF1.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_2597.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/40000/D6872DA6-475E-2F4A-8553-A4D8CC950B78.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_2590.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/40000/7682D50E-7999-E44E-8412-2AD2B8457A97.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_2591.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/40000/CE5870A2-4C2B-E242-83EF-43B9E861D4B8.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_2592.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/40000/4E0215AB-7855-784F-B490-2199EE765D56.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_2593.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/40000/5CA558E3-8609-7B4D-839F-03651A362984.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_2598.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/40000/307A12F8-2EA3-9F47-B618-812046EB2C79.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_2599.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/280001/CB3D0111-D5FC-8D43-B836-155A76CEC2A9.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_1876.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/280000/DDBC7A79-AD52-2C47-9A64-F22E665FE771.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_1877.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/280001/7EE34C24-3924-F044-AA3D-F265DC75DD75.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_1874.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/280001/E9DCED3B-ABB0-FA49-9365-B4267962741E.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_1875.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/280001/15E55089-2571-A24C-BFA6-AF1F73EF7D07.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_1872.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/280001/C207A4C1-1CFE-1944-89DF-C966343B8AD5.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_1873.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/280000/25AB5BE7-3B24-0E4D-BB6E-3DB31ECD51B2.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_1870.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/280000/0B7251EA-15C0-804D-A4DC-D7BE764218C2.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_1871.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260001/5F82604E-D5F7-994C-B384-67B6D6B7E237.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_595.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/280001/BF2B079C-86A0-6546-94A8-4FB48D7C6CA9.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_1878.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/280001/59E6E7F2-9CFC-874C-9AE1-446E36E4B8EB.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_1879.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260000/12D307D2-5B30-BB4B-8919-6F6EC025828C.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_977.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260000/61F8DD66-C505-134A-AF33-393F279B041B.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_976.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260000/08B336D5-3BD6-CB43-9B87-311488546C76.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_975.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260000/D75BC497-84F8-6A47-B5D5-A8C7BEB8B02B.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_974.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260000/87321272-ED31-024B-88FE-B3D61FA37A60.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_973.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260000/09C67C38-22ED-BB44-A84C-DA1E8AC42EE2.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_972.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260000/3AAA38FC-4939-4048-963F-C16C4453C25C.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_971.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260000/0F6A9A2D-5B70-9341-B500-E89A76BCC35A.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_970.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260001/3E63411D-488D-B14D-B21C-F442832EAAC3.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_596.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260000/20F2D25F-4E4E-A242-8596-12EE1E7A84DC.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_979.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260000/682A994E-165D-CB42-B1A5-4BC83824207A.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_978.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/280000/814A6770-281D-F946-8528-9F1C227B5EA4.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_182.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/280000/E929E205-E510-C849-B79D-4D216A58B3C3.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_183.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/280000/E8C070C7-2E03-844F-9AA4-261EF00290C3.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_180.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/280000/EC140275-3640-D94A-A962-92318DC93823.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_181.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/280000/4BB26A50-6E56-B74F-BCA9-DBD153B08D5E.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_186.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/280000/83062A8E-A494-2943-984C-205FEDD17BE7.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_187.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/280001/1BA4E758-B6DB-5141-88A9-F7CC73DA2089.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_184.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/280000/18C4FF64-3D50-F041-95EE-128E99E541AB.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_185.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/270000/F743EBE7-4301-2A4E-99A0-AA0EA6B1DF8E.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_2110.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/270000/8D3F290E-CC30-8347-84E3-CE2103C9F516.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_2111.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/280000/EC4166FD-CA3C-3E47-9FFD-3D45399FD65A.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_188.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/280000/93E962BE-9502-C941-A187-871B07FF90EB.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_189.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/270000/8267D534-C1D6-A542-A95C-40A0E43318BD.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_2114.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/270000/BE6942BF-339D-B740-B026-83BCFC6C2730.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_2115.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/270000/8F30FE6E-6168-C54B-B0A7-63257A12D8D5.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_2116.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/270000/81075E90-291F-254E-8CA8-ABBD1FE8412E.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_1559.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/40000/256F4952-7B2C-7C43-8420-AC738B50C278.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_2622.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/40000/AA2AA9DE-4135-B84B-8DA1-8178D90538BE.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_2623.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/40000/73964447-9A5C-B84E-BF07-497B17BD465A.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_2620.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/40000/2446C869-756C-CA40-B9B2-58D4F4A449F3.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_2621.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/40000/C0C3A00C-B3F2-EA43-AAAB-4A45C03009D4.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_2626.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/40000/44D94559-5840-B345-A065-AF71DFA13073.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_2627.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/40000/FBBB5420-FB89-9142-B920-E20EBD385793.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_2624.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/40000/D4FA084A-2429-FC44-B936-D51E8285041D.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_2625.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/40000/2838BCF3-0BC7-FA44-8DB3-4B45F6097D00.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_2628.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/40000/48FB1208-354C-CF40-B781-2D511DDA2610.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_2629.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/50000/C24FCE4A-5B17-0242-94BC-4D462A1BBA04.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_1919.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/50000/BE15285F-BB45-F541-8607-A7B20F610806.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_1918.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/50000/46D9E767-4F53-6847-82EB-53C61E29DEFF.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_1911.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/50000/4D2690DA-2914-974B-9DBB-82487184FBEB.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_1910.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/50000/7F2DA70C-6F92-8E4D-9FFC-D8990D6C0BF1.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_1913.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/50000/F14D6F61-1FA9-8F4F-9037-4F188C3E90BD.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_1912.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/50000/B522B116-C647-0A48-A119-CA732C54DAE5.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_1915.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/50000/8FEFC572-968A-5748-99F2-F08196BA8DAE.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_1914.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/50000/9F51A4C1-6412-714D-BCFC-1D36D0B69001.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_1917.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/50000/5F29E89C-DF9E-494A-803B-233CFA8F0FE7.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_1916.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/40000/92C01191-43F4-534D-97D2-079EF70AE430.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_869.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/40000/C7937292-415A-AD44-A311-D340FC2C6D01.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_868.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/270000/E48359DB-D6A4-A541-9A10-1F8811D706DA.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_2113.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260001/8D356F5F-F1C9-C140-8B4C-E20E7B83F58E.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_516.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/40000/072F41AB-543E-A646-9A5F-4E02991F6329.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_861.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/40000/BA75349E-DC13-4E4F-9CF2-DBBEF93EF3A1.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_860.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/40000/E43FD9B7-1BEC-474B-893A-590DD63A02B7.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_863.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/40000/D08B1C9C-15EA-B145-B95B-DDA7BF406CAD.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_862.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/40000/0FE01831-5C82-CB49-BCFA-329645829E50.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_865.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/40000/CD4BE471-573F-6740-8F8A-6CA539C45045.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_864.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/40000/995206EA-9BAE-6D49-9AC6-A0B0001309F3.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_867.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/40000/10D72A76-873D-E344-8AD8-DA6ED5AB8BF1.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_866.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/50000/B0600705-0EEA-4B46-8433-772171BFC436.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_2024.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/50000/3F0449D4-28BD-CB4B-8D8D-296803F21D87.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_2025.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/50000/3C78A06A-16A2-464C-9798-172ED1462A25.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_2026.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/50000/563473C0-CA49-A041-B674-E0B2FBA34F04.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_2027.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/50000/A9912544-3326-8342-BC41-744D2B88B1CA.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_2020.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/50000/53704245-2ACD-F74E-A964-EDC4B112EB24.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_2021.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/50000/2507F68D-C21D-5B4E-9F8C-02B3F585025A.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_2022.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/50000/91DF19F4-3299-824C-B956-557A51E979E8.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_2023.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/270000/4D728EAE-7F1A-E341-9430-99DBAFD4DE31.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_2117.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/50000/8103638E-2A4F-3E4A-8CEE-D9E11EA31C0E.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_2028.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/50000/32008C03-1714-8B48-AF94-04B66A8B9C78.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_2029.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260001/43D335A4-06FA-9E43-B599-FA9B9A2FEE26.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_656.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/40000/05A5F8E9-3BFD-6244-BBD9-DE36B48DCA0D.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_883.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/40000/80DF781C-5ACA-C742-9F92-61645381FAF9.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_882.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/40000/F3904FF1-955B-9141-8AE7-AE51355BD189.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_881.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/40000/878D72DB-98FD-2741-A955-13E433B51721.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_880.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/40000/31723049-5219-1943-89E1-FDD7E81FC7B5.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_887.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/40000/FB6B81C2-79EC-7142-83C0-087A4A960248.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_886.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/40000/4971FCB4-95F4-804A-B564-9F9C6095F35D.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_885.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/40000/FCD2CC72-7D0F-AC43-B14E-EAF105E0EB29.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_884.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260000/6C6E7185-4078-8D47-9710-725DC750B15B.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_889.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/40000/60768C8A-5D6C-9E41-A9CC-2937A36C344B.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_888.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260001/C964E671-EB8F-0F41-A7DE-D6B2888FDF13.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_657.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/280000/1A65A78B-0924-764C-B33E-39A3E2539546.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_63.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260001/EA7D304F-5B79-F34B-A515-1E643132F566.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_775.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260001/E3766537-6E23-0D43-9449-B072F9C13589.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_774.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260001/2D74BFB0-22FA-1642-A18E-E9F931B69AF4.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_777.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260001/831B8685-44F1-0C4F-A398-AF9E64C80888.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_776.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260000/B0B53FB7-72CA-6748-9373-455F9434B776.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_771.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260000/0738A4E6-52AC-3142-9CB7-4E059E55D4B4.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_770.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260001/8B7AA686-83A0-964F-91CA-0FB0F4DD4B2B.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_773.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260000/318D03A5-939D-A34E-9266-1D99E0B76CC7.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_772.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260001/82EA1216-EF62-E54F-ADD0-638EA1FB5383.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_779.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260001/56699B84-AF0F-BB4C-9E0B-AFA98D31175A.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_778.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/280000/4B312BAE-72DC-2847-A47C-F037C35CEE46.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_77.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/280000/815D1E7B-2CDF-C542-8036-153EAA67C181.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_76.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/280000/4CB32263-7C0A-2F44-9D4D-F671C6AEB09C.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_75.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/280000/1162C414-177E-8045-944C-517E944709EF.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_74.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/280000/57FE4D86-643C-BF4B-93EE-60CC94A9EA30.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_73.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/280000/E430EBD4-51DE-F543-A0A4-B334A5263491.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_72.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/280000/21379DF6-1B31-0647-8950-0C4B7CAD509E.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_71.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/280000/BDDC952F-939D-0F42-83F6-6446654ED661.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_70.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/280000/1D08CDFA-1048-F84D-8F09-E5F71915549D.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_79.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/280000/6F59C616-93F6-DC42-9775-70510298790C.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_78.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260000/23768FC3-1DA4-C649-ABCB-BFE428FFAADC.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_1043.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260000/954CEE0E-89C8-4C45-840D-4E9ED9861B7B.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_1042.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260000/B65F479D-CC5F-6B43-8E9B-638E92EB3552.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_1041.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260000/92B20647-E828-F840-BEEB-23B05E32971B.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_1040.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260000/0CB669FF-60C0-F84F-8165-6D75030DE776.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_1047.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260000/49CA323D-1081-C64E-AA1D-B9D7BDCB6277.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_1046.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260000/667EE1EC-2E13-C548-8280-BEC619DF3D76.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_1045.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260000/D95A1B56-FC30-3B48-83B1-9941B1C19B4E.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_1044.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260000/28BC1952-0AD3-F94F-8DA5-5003257CBDDC.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_1049.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260000/DEBE6246-10C2-D04C-BF86-D95912233A5B.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_1048.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/270000/7C1941C3-003B-0846-B712-F3CCECB97FE7.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_1681.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/270000/26E2E6AF-9602-E143-A21C-0D5432848FB7.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_1680.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/270000/17B38E27-EF0C-9240-A8BB-FC53FFA74A6C.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_1683.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/270000/4AAFA490-5830-074F-9A0E-9353CDBBB3DC.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_1682.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/40000/B9608330-68F3-594F-99DB-7F35656DAF6E.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_1685.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/40000/C41E7EE9-0C90-D541-B8FA-941855A8ADB7.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_1684.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/40000/9AE63908-A288-E047-90D1-22BE627BE328.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_1687.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/40000/96E2EAB5-3D09-594B-AA3F-A0ECD9FB8B6A.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_1686.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/40000/AE44AB13-750B-1A4D-ABF0-704B38DB3096.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_1689.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/40000/9FEBA846-C70C-0E49-8E77-6FC20FFBB01A.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_1688.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/270000/CF6D0222-2D8F-4B41-8113-9E443BB3ACA4.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_1269.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/270000/8925FB54-2BB9-2143-9693-ED4CC93A7045.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_1268.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260001/272E8AEB-0ADD-3C4B-9303-CC2228B64F4C.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_669.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260001/59E05D1C-1211-8E48-BAD0-EB6D0378FF24.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_668.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260001/91C1944C-49D7-F840-A1E8-72BAC85646BD.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_667.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/270000/C3D1E51A-FEED-C84A-B821-331EDE9FD61D.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_1262.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260001/B86DF7E0-6159-2E4E-B8F8-D3ACB80ECE4A.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_665.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260001/530E6656-7D3E-714B-922A-2514163DE195.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_664.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260001/E70655B2-05CD-A14C-B672-18263EF9EB59.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_663.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260000/368DE0C6-3D5A-8D4B-BF37-D36EAF58E5B5.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_662.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/270000/7872C052-1D2A-FB48-94E6-FDE142B5AA6A.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_1265.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/270000/AF428E02-DDD7-8446-B8B6-D52DAF633A97.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_1264.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/270000/3B9AF5DB-3DAC-834D-99ED-469E0A847832.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_1469.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/270000/8F593E24-0FAD-BA4A-9DE2-33D33778A18C.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_1468.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/270000/2DE526A4-6FC9-B648-AEDE-26614780D26D.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_1465.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/270000/7DDBAEE5-CDBF-2045-AC99-CCC34692FAF6.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_1464.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/270000/DD338717-4525-0849-9AB4-F814FAF49C49.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_1467.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/270000/B1FEACD9-5568-584F-846C-EBEB25EEB98C.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_1466.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/270000/99584E86-2387-0742-A871-1119A76A0495.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_1461.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/270000/2D647BB1-F9A7-FF4B-9AB0-2C1034D7493C.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_1460.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/270000/72F7BBFD-71CC-3C41-8511-4BBCCE899589.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_1463.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260000/9DF279C0-96C4-3A45-BC43-F3FEB24F131F.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_1019.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260000/44AD0957-8562-3E46-970E-CD0F47976EA0.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_1317.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260000/46C53927-68C5-7341-94E5-DBE289F9A763.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_1316.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260000/88267F07-DE16-0B4D-909B-10E36B8C6BF2.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_1315.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260000/82B90808-2EE4-C445-BB7F-8EDC2924F31B.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_1314.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260000/7E9699AE-99CA-F746-9D8E-6F519762F912.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_1313.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260000/8DD6D333-84C8-7B4C-BA84-777915BD9EB6.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_1312.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260000/10971D17-17A9-2B4D-A09A-1DFA6723778F.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_1311.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260000/9FC2C207-D232-DB4E-B7B9-B9848E4FBE56.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_1310.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260000/13892169-82A0-8F47-9902-0BDB77FE1B08.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_1319.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260000/D0142D32-E5A2-0542-A728-3CD16C75831E.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_1318.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260000/5E34DD3A-930F-234C-8C1C-91F3A599F768.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_1010.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260000/1ABFB17A-56FE-C84C-AD16-F8395C222A9E.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_1011.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/280000/C5549D9B-15B6-8F48-ADC4-EC897E1D826C.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_319.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/280000/70B382BB-4BD7-7D40-8754-6922C9E6E209.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_318.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/280000/B56EB288-0C04-A949-BAB1-C6BC15287FE5.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_313.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/280000/C5700064-456D-EE47-B0BB-C641896D9618.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_312.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/280000/3CEED7D9-83BC-6148-BD0D-4DAEC84C2E17.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_311.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/280000/FA73EB07-1B51-ED41-B1A7-34D97C748B07.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_310.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/280000/3E79A02C-FFF0-3A4B-938B-FF59072CBB79.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_317.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/280000/89569877-E961-6648-8387-705F1912F2D2.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_316.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/280000/0FA87CE5-7247-744D-8954-2D4E5080F18C.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_315.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/280000/E2DCED54-B75C-2840-AF19-2F0E8055FBBA.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_314.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260000/9811587C-E727-1E4C-9A71-DF5F3DE2EC40.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_1520.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/280000/7161D9E4-69DB-DC48-8AAE-063D98071BDD.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_443.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/280000/38052892-CD88-7847-800E-AF0830C0C1F9.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_442.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/280000/7FB30797-AE2A-654B-82A8-D99B749EE24D.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_441.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/280000/AC69751B-4EB8-DD47-9291-68B13C95D145.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_440.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/280000/DF48DD6F-333D-0A40-8D42-0EDA514F9BB8.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_447.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/40000/7C614F3A-D08F-FB4C-8F60-4FCCFFCAEC24.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_2698.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/270000/8611AC67-4748-3248-9DD8-9D942A698725.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_1330.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/280000/80F1ED3B-45BF-BC49-8FF8-242B44190E50.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_445.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/280000/9F4A845D-87D8-2E45-9C3F-6866884211BD.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_444.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/40000/AA61B9A8-7668-1942-BE62-C02D77166B68.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_2692.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260000/9E9F9CD0-4EDC-5444-84BE-B0AFE75A3324.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_630.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/50000/2005CFA5-EC14-4D46-9B95-DD3D1687F67D.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_2299.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/50000/20BA5427-A308-6546-B611-F4843E44E46B.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_2298.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/50000/B063A18A-9211-2E4E-BD9F-C3E71E2E3429.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_2293.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/50000/55E0F360-F11F-C942-8C7F-0A3E2729B73C.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_2292.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/50000/6DC4FF89-61A6-9046-9AC6-F378078519FF.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_2291.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/50000/96396A71-95E8-B547-937C-C736BB9EBF1A.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_2290.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/50000/5A065A6C-E227-B044-B5F7-E937E2D9C012.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_2297.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/50000/4BEF8C9E-DA3E-694B-A0FD-C3141D07C20E.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_2296.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/50000/5DCACD2D-4279-FD4A-906E-B0A176FF8B90.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_2295.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/50000/605DFDE0-E68C-D540-94F2-7533322CC350.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_2294.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/40000/00C1EDC3-EB59-1A42-AACB-986A4C29A532.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_2569.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/40000/16E21A34-3A1D-AA43-9467-FDEF31C72993.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_2568.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/40000/BA1CEE94-2231-BB40-809B-FAD6777A5F9E.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_2561.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/40000/2F240699-33A2-D443-BA4C-EE3951F48BAE.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_2560.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/40000/55966888-2A93-6241-9E91-6D28ACED3077.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_2563.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/40000/F64FD333-6D3D-4845-B07E-DFCFC6371810.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_2562.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/40000/E0E54ABC-3755-914E-AF0A-A5F5B4321B74.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_2565.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/40000/CB15E64B-FFB7-7042-8EEF-CD91F570C176.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_2564.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/40000/342E63A0-107F-A245-8F1F-7ED91443DB07.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_2567.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/40000/38809D15-BF38-1442-A003-04B4252F62AF.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_2566.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260000/9F8197C8-2436-B549-83F8-850DD5F404A3.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_908.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/280001/91CFC55F-9B93-7442-9234-AFCA4F6728F5.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_2369.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/280001/6503E54F-DBAD-654A-A010-FAF142FBBE03.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_2368.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/50000/A430CE13-2D2F-4E4B-90AD-418CFD97A3D3.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_2367.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/50000/3BB40039-92D3-3843-A6BA-8EAB149B21E3.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_2366.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/280000/A2E06B05-A3F7-F849-909C-9233E9047949.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_2365.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/280000/BA9BD2B3-539A-EB4B-BC55-7AD577127CBD.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_2364.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/280000/CA235C5E-E7CC-A540-A899-94A42A5B275E.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_2363.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/280001/02F4D36F-E51E-4243-8445-5939BF648ADD.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_2362.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/280001/D5C5CCD5-B4F7-414B-AB95-7FF420E6EBF9.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_2361.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/280001/01E57910-6F45-C841-A1CC-3AB94A6C7404.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_2360.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/40000/9D581140-EFC7-9D49-808F-80B37919AF64.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_2587.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/40000/3DDB4F7E-D868-214D-9302-4F3C7554F054.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_2586.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/40000/EEF24686-E74E-4548-9987-2B204B01FAD0.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_2585.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/40000/3686BEAD-EB41-2447-8046-1FBF2211223E.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_2584.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/40000/56B4A9A9-1C10-6145-9434-C0E0F3EB57EF.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_2583.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/40000/61D32BEB-0572-CE4F-90A2-7727C5798005.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_2582.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/40000/A3F090FE-4155-1946-AA24-3689FDDBC0E1.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_2581.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/40000/F92A4EF9-DC7D-C040-BA8D-79A2D9D9EF3E.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_2580.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/40000/AAD3D900-EDC8-E341-BDB6-22E234B11162.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_2589.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/40000/05F631B1-475B-214C-B1C5-610BDB80708B.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_2588.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/280000/4DB78BA2-1F7D-9842-B243-A127EF6B0944.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_1861.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/50000/4AC089E6-294B-F148-977C-A7853D05843D.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_1860.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/280000/48A760A2-AE7B-7D4E-8E8F-960F4A7C0C38.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_1863.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/280000/430EA7A9-7D28-C64F-949F-8D6B07BB20AC.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_1862.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/280000/34C2C6F8-42B1-8543-9E61-2F38B1E26922.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_1865.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/280000/1CA1F9D5-0757-B24D-A4A9-7A111EA691F0.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_1864.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/280001/C03A5E0D-B969-9D4B-8D05-1CD1E85BBE8F.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_1867.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/280000/9FF34E9D-D2F9-A747-96F8-3A60A814C5B5.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_1866.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/280001/8AD663F2-10BD-164D-B480-86CF772A328C.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_1869.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/280001/1DE9832D-9E86-8041-8CEC-18B5A5E4C60E.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_1868.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260000/3080C750-0FDD-6D45-ABBC-B2A8DBEBD18D.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_964.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260000/F1F9015B-9455-BC4F-A2B3-04B924D0A851.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_965.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260000/25F9E3E7-895C-1E41-B100-96896ADE0F46.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_966.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260000/DBC1B6D3-1276-FA4E-B999-B30E5EEE5FB0.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_967.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260000/1D70B9F6-E05B-4242-9070-121315A2E3E4.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_960.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260000/F8B6B6AB-3A7F-3A46-A3A9-8209DD41E7DE.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_961.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260000/CE5F82C3-349E-7049-8389-ECA17731B881.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_962.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260000/3DDF2D01-3B85-934C-9A81-3662225DEBE4.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_963.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/280000/259687D4-895C-9140-BDD1-95C07B481FE1.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_401.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260000/7A75A6FF-2C24-B347-97ED-DD669983D441.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_968.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260000/B98117FB-4444-F84F-83C1-B825A1A77C4F.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_969.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/270000/77F6583F-7149-DF45-8FCC-58F95FFD02D8.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_2109.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/270000/6B5CD6BC-1FE3-724E-B1AF-AF9E7D2DBB92.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_2108.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/270000/683CA206-44D6-FA48-8475-8F9CE032AEE7.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_2103.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/270000/8AFAB6AA-CAE9-574D-8135-AA1EA3AF00AB.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_2102.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/270000/C08A9B63-4FB6-354A-8C95-D1E647DD4239.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_2101.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/270000/D3CA06A4-88A6-C144-884B-6493D0EB40C7.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_2100.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/270000/C07D3A0B-5FC1-694D-8F0A-3B6AEBED7C03.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_2107.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/270000/E88F349B-B907-054E-8E5A-56D008231C8C.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_2106.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/270000/3E7C3B35-2ADC-B743-BBE6-32AFD42A2BFE.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_2105.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/270000/49BF07A1-3FFE-8142-B7E8-F221313BD543.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_2104.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/40000/6BCA8A4E-9770-3743-AA7D-7D89236A2EB6.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_2639.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/40000/6CCC2B9B-08D6-6446-B0FA-556930A40726.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_2638.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/40000/B7AEA646-F2FD-4C42-9F7A-B3D27B316638.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_2635.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/40000/9C4622D3-9C13-B242-B24D-3F94D90FAE5C.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_2634.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/40000/F7AA75A3-18D2-FE41-96BB-4B8D082472E7.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_2637.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/40000/C6F6102F-0D91-3241-A8F9-02DD3111379B.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_2636.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/40000/683EACD9-5ED9-E34C-BB0E-EF22A10BBDA5.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_2631.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/40000/67BE1942-BB02-F341-B458-9DC5E962A5F3.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_2630.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/40000/8EA24949-37EA-3447-8C50-18BF4860DBD4.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_2633.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/40000/E3C6A6D8-D6E4-FA47-8265-E1ED86D431EE.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_2632.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260000/42B529E6-AB2A-5C41-B304-802DE7164782.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_1537.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/40000/0B96E274-AECE-764C-8A5D-5F1BEC8C315C.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_878.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/40000/0BB75153-71B0-F944-A80A-022EDCBD0543.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_879.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/40000/042C8FB0-9FBD-424B-BFFA-232027FFB8C8.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_876.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/40000/9630E67A-300F-6641-940A-843847688544.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_877.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/40000/E086FDCD-670F-464E-A6B8-045D6041471E.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_874.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/40000/86061D43-7775-C849-8454-55015BE61F03.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_875.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/40000/92579CE3-8B34-694C-B6BF-97DDF0608E15.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_872.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/40000/36832A13-8B8E-874E-8AA0-5F4F05CF1177.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_873.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/40000/C097EF83-CF2A-6649-BC9E-172C9A612DD0.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_870.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/40000/10B357F8-3896-0446-9110-AFCD59F09852.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_871.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/50000/F82A6DBA-E77B-5048-A3F4-2E4E3EFAC4A7.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_2037.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/50000/902C395E-8032-9341-9AED-2FC3CCE7B859.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_2036.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/50000/885DDB81-D5B5-6345-A83D-F251FD528168.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_2035.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260000/1F0E0D60-BC78-104C-9F53-3880173B1009.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_1242.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/50000/A56524DC-DD30-554E-B1E6-EFBFD2EBC676.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_2033.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/50000/9CFE07F3-E7B4-4645-A677-38B353D80981.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_2032.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/50000/C57F0066-2938-1849-AA60-B26A0C73DEE3.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_2031.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/50000/3DAB978F-759E-8D44-A76F-7DD344BE4F82.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_2030.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/280000/39789092-8B9A-6646-9031-A52564DCC2FF.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_9.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260000/5ECA39C9-A8A7-7F47-8666-13135E0013D8.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_1533.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/50000/62CC1648-0C66-9C47-9165-1730D7A6310D.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_2039.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/50000/F8F3DF15-8887-FA46-AF04-2103DE3EE507.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_2038.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260000/27850801-0BEF-084F-9405-378B8AD5490B.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_890.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260000/89520C16-6441-F346-BB70-D5ECEF9361DB.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_891.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260000/76CF3F3A-BF3F-F74D-B0C5-0535B9A8E3FC.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_892.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260000/0FA1FA0C-7809-5344-AA5B-C429E07AE426.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_893.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260000/5E70DEC6-1BBC-9A4B-8621-38CB85301089.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_894.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260000/1E0AC27D-723C-5C40-A915-124C0D802C8A.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_895.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260000/6D5FD4CA-BB0F-DB44-8BE3-0CA980623209.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_896.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260000/0388A30D-E3A0-2947-8F6A-52B8B3F4C381.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_897.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260000/2B59A428-76C8-4940-AA5C-B6BC7C450B2A.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_898.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260000/059EA86A-6D43-074E-B511-1B33CFBBA6F3.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_899.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260001/601A2B61-B1C6-9F4E-AA01-CDEDB0CE8FC4.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_646.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260001/28D79186-5AD9-1E4A-BC4F-1FFC918BCFE9.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_649.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/270000/E600B921-54C7-1948-8D5E-91BC9E1EEEC1.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_1248.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260000/32C38877-64C7-D54C-92A9-AAAEF61D329C.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_641.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/50000/3702D1A7-9816-C347-A97B-DB22C4C42290.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_1964.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/50000/5C1F4127-647E-5649-9A66-FEE73747C89A.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_1965.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/50000/E9BA5528-420E-094D-AD26-9D38CC36C0F7.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_1966.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/50000/C892BCA8-5F3F-DA4E-888C-74B062EEA8B5.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_1788.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/50000/DEE656D8-B78D-E949-9A1D-44826CEF4055.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_1789.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260000/9D13E43D-44A4-E94F-9DF5-7B80786B030B.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_768.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/50000/11D6D1C7-DF4E-CE4C-9CB6-57DCDCAA59CA.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_1967.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/280000/AA565D42-AD3B-D740-9EEC-67CA31B01F68.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_265.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/50000/8361EFBF-468D-2C4D-99AA-5904101B690F.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_1780.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260000/C445EAEE-4D2B-614C-8A72-F1776AC08EE8.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_1612.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/50000/E26C9803-1184-4045-817A-49EFA2F98785.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_1782.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/50000/D381CC2E-F9E1-A347-8BAE-F172B04B5BC7.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_1783.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/50000/5426F746-FC79-D546-943C-376510A9B151.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_1784.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/50000/2929FD50-5441-AD4B-A298-5712A357C7F6.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_1785.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/50000/D67646EE-E5FF-7249-8F86-8273B3DE7F87.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_1786.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/50000/14E10F09-2DF1-BB40-8964-FDCEAC006935.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_1787.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/50000/D01110E9-36F7-8249-BD0C-80488CE88949.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_1962.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/50000/FA1B1AE6-C1A0-0C47-8242-E559E567EF42.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_1963.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260000/F273FEC3-2374-8149-85BD-20BBDEA8F9E5.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_1078.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260000/2923FF7C-4905-0C48-A7E9-4733A6E3E756.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_1079.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260000/43D90564-F4E1-9443-AC36-5FD42E99F125.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_1076.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260000/5011A580-6FC2-AE45-BCF1-F89C2DF97A10.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_1077.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260000/6172C5D4-F539-F34C-8110-3AC4F6C570C1.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_1074.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260000/5DA9B8F9-5F12-D840-8707-E9F81807594C.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_1075.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260000/EB2957A3-2CCD-6444-870E-B19C617ABC81.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_1072.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260000/B49F3F3E-8C1B-A343-AABF-C1D35BDA042A.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_1073.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260000/C1D78DB3-CDE4-6B4D-AC2F-9B8D4BB8CD97.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_1070.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260000/70449009-64EB-D048-99B0-14D231F69635.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_1071.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/270000/9837BD61-8E5A-9641-8340-DE655C592394.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_1678.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/270000/9AFA786E-0A5D-424A-8C57-331D39D3891F.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_1679.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/270000/DDC0392E-5B83-B246-A308-3824EC006223.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_1674.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/270000/631DFB1F-592E-4F44-B499-7AB0E76257B1.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_1675.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/270000/F15A3111-E23B-404F-B695-D62960D45CEB.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_1676.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/270000/898A97C8-D1C0-3941-BD92-EFAEB3B98944.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_1677.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/270000/6EA875B5-5546-DB4B-A2D9-FFCCA84C6A01.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_1670.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/270000/5DDE5721-6DF1-9B44-961D-A368842D3652.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_1671.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/270000/E1215A51-0812-EB4F-8AC9-AA0623F12A5D.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_1672.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/270000/6665CAF8-7785-9949-BE30-AC21C039AE4A.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_1673.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260000/0E7EC6AF-3277-CE45-A609-B7186BC7FB7C.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_1094.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260000/1CC9A2EA-B31D-0A4B-A2E2-8B183B55A44A.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_1095.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260000/99FC5E60-26B3-3A43-A7AA-0D57509F1B88.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_1096.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260000/5088E564-2AD8-7A42-88EF-F4E4C4E57F3C.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_1097.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260000/9AAD0599-185C-514F-AE9D-746EE8C44387.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_678.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260000/6E746790-85B1-3A4C-9875-B2C15BB31D15.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_679.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260000/0886CEBB-2006-3140-AFE1-722ACD2E79F1.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_1092.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260000/1A2DA732-5CFE-4547-BD7B-331A466DA52C.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_1093.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260001/32E832C3-B052-054C-8C4B-A98FE5BB3112.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_674.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260001/68E23ADE-844A-BD48-8D98-1DC4389A146E.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_675.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260001/B93155F6-0C35-6A4C-AF6E-AE2D17A020AA.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_676.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260001/8FA569E4-59C9-C74A-A270-BBD78FDF471D.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_677.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260001/00B1861D-877B-FD49-929A-85DF2C6F16F7.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_670.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260001/3B361930-A11D-E44A-A3DA-82D0D6474148.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_671.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260001/92425A6B-6BCE-3C4C-9B55-63136BD2462A.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_672.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260001/1DA0B660-3906-9C46-B730-6646085933C5.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_673.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260001/6A3DD63B-7560-5F43-A8A9-E584D9136414.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_645.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260000/6D4CF5AB-CDC4-1543-87EE-F55DD58EF51A.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_1418.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260000/6BD5E2BB-8273-8A46-9103-1EB786624278.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_1419.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260000/A2186E13-A0DD-5E44-9769-AF257E6763D3.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_1410.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260000/39F9EE2B-8BE7-F547-8983-0BDE25D8B5E9.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_1411.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260000/E01BB300-4D86-3C48-AC9B-F9133FF2E5F4.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_1412.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260000/294C1F4D-90AD-EB46-9C83-2C2873E57E72.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_1413.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260000/5AA47F92-75DA-3F46-A660-F02DA594C37F.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_1414.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260000/299A955F-1E53-CD44-80AA-D0F2E8D156C2.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_1415.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260000/E30CCFA6-BD18-3A4B-A459-A1D4DDE648F4.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_1416.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260000/EB5CA75A-930E-094F-B2DA-B20373A98DAB.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_1417.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260000/00F0369F-B9FA-5643-B0F4-F286B80B30B5.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_1322.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260000/6B14519D-996B-1145-A0F7-3736BB1DE96F.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_1323.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260000/231F62EC-457C-AB41-A584-71EF355550BF.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_1320.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260000/4C548C14-9C6F-6D48-84D5-839A2D1BA8DE.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_1321.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260000/A3DF1E71-50B4-6141-9664-173DC32321BB.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_1326.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/270000/4FEEB707-A1ED-FD41-A388-A17242444477.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_1327.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260000/A665C2C1-7BD6-F14D-8F00-0E61F0FEA3E8.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_1324.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260000/4D452CC7-A630-F247-A677-5ACF31B4FE44.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_1325.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/270000/D73E1C42-2210-ED4A-A620-95D03E7F90B7.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_1328.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/270000/21E5675D-6430-7640-94A0-B92FF4AFD7FB.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_1329.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260000/505305E7-D43C-0945-931E-7A9902C0CD4F.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_1531.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260000/A65B3FAC-683A-0F45-83F1-ACD66683BC0B.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_1524.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260000/F8107990-CC9E-E845-A3DA-1DDE2D037745.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_1525.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260000/A3BF1463-C3EC-F648-B018-A75CA05DF519.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_1526.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260000/2270953A-B422-EA4E-BFB0-B8138A19102C.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_1527.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/270000/57CF7110-4F67-5540-92D2-F6605764C625.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_1252.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260000/7395248E-BBD1-8C45-8386-55846E825E4A.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_1521.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260000/39449B6E-532A-8C44-8C64-E2AF547A614F.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_1522.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260000/470A240E-C657-0545-BBB5-8CCD234BFDBA.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_1523.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260000/9D3178F1-D762-5742-B558-3C052CDEE5C3.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_1528.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260000/82069161-635B-A847-8839-DC3F1E1602D4.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_1529.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/270000/8590BB8D-694A-E74A-928D-64A1C8DF3814.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_1258.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/270000/0EBF20F8-1B92-F84C-88E7-431DD992D085.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_1259.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/280000/EC9F7F3F-96E3-4545-A983-0269793DDD93.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_308.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/280000/F17109C0-C17D-5C44-8CF7-F73731DD2FA4.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_309.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/280000/85126006-9C98-6A42-BFD5-9927833E44E7.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_300.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/280000/2E1734B6-FF19-864C-B194-58CDCF961EB8.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_301.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/280000/169D5A44-121A-DF4A-AD1F-49BAADE79428.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_302.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/280000/D19C893E-EFAA-9E44-8783-A1CFE2404D89.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_303.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/280000/553DAB02-2F8E-9A4A-BBDC-5B8831638016.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_304.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/280000/71AEF994-9434-944E-B07E-64BA343A4300.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_305.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/280000/56D48684-E3E0-9E45-A2A1-337AA37C5559.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_306.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/280000/1E0A92F2-4B48-0A45-AA1A-9C026C9EC4A6.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_307.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260000/11387134-D37D-CF46-9522-C12C46A3F613.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_988.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260000/E3213317-16A9-F14B-A333-5A2ECEDE34B9.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_989.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/270000/4A393472-CBAE-1A40-B6CD-6E762EB5E237.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_1447.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/270000/91221399-145A-E744-B564-8ED24C219CCF.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_1446.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/280000/41727263-DDDD-3C4E-8A8A-02347CB0A7D4.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_473.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260000/29295E87-9078-4E4A-B717-C8D399CF96D3.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_984.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260000/A6B8344C-0D2F-094B-B1BA-C68EB8D11BD6.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_985.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/280000/F7A6B2D4-61AA-4942-9E9B-00B962E8C627.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_114.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/280001/60D89FCF-6A9E-2142-90F7-277F6EC151BF.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_2358.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/280001/62CE6813-41BE-6949-AABA-5C06666BF8E1.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_2359.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/280001/3AA8133F-07AB-AA41-90E1-FA4395A95C71.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_2352.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/280001/4328B453-1CC4-4D4B-9DE7-5981D5E6FDB7.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_2353.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/280000/2E09AF76-EDAA-174C-9270-BF14CDFF5B25.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_2350.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/280000/4943A16B-C2C1-6F4A-9A9F-3179AFFDF85D.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_2351.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/280001/0BE235EC-B1C3-1D4B-96B4-ADCDC00C11F3.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_2356.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/280000/02A73F70-B2EF-2347-B7C0-AB14AED775A9.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_2357.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/280001/3AF2C314-8CAB-8D47-9C83-9532C9E01BBD.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_2354.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/280001/F2F993B0-981A-9645-AD38-1152C0C521C7.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_2355.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/280000/319C29B0-3231-B34D-A207-9F72D7F2BE0B.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_112.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/50000/587AC884-F5F2-3643-9527-C29AD261B214.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_1898.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/50000/6FEBC5BB-DA12-4C4C-B2E5-09397BE47251.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_1899.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/50000/B709CE56-0210-144B-978E-E6C7FB055034.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_1894.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/50000/7E8EEA89-4477-434A-A8C3-7E07D523E9C0.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_1895.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/50000/AE42EDAB-D95D-2941-947C-07A1EA1A7408.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_1896.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/50000/E4603076-B013-9C43-BF45-68B0248CB3B1.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_1897.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/50000/C12BB5E4-D7DA-914A-85F1-61A89FE6FDDC.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_1890.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/50000/1C822160-887E-2D43-8328-25397525462D.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_1891.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/50000/2E453211-533E-814A-B583-AE89AC6ECC75.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_1892.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/50000/6B01BFEC-93E9-7F41-A6DF-2EF827F5C2F9.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_1893.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/270000/D4022E03-FA76-ED48-A533-8B6E401FC760.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_2136.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/270000/7F6B3EFA-B667-AF4A-A6B0-F7B50237F524.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_2137.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/270000/0E68CB9F-C89D-9141-9AB3-250B6BD88B1D.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_2134.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/270000/AFE3CCDE-5147-154F-9764-6AB1A1FFD298.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_2135.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/270000/4FB3C373-112D-7C43-B924-15670B0D0CCB.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_2132.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/270000/5B62E7BE-8A2B-1847-9964-00F999FFFC24.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_2133.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/270000/71A047D7-47B4-4D44-B256-D81CAA50EFB5.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_2130.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/270000/647F3816-41EB-A040-8A77-2A09D3B6B6DB.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_2131.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260000/648D634A-6F1A-324B-A323-263E69E064FE.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_951.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260000/89393002-345D-6D40-B8F5-86949E93FBF1.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_950.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260000/7B77A501-846C-D442-886E-3ABD63AC88E6.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_953.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260000/80A3F4E2-4542-604C-88B3-5F05E12BA80B.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_952.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260000/B06BE9C2-823E-A347-AEA8-93CC7DD231E6.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_955.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260000/40401D9B-7B21-B145-8862-975017DF6730.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_954.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/270000/AAFB01DF-89C2-9646-8CE9-E9DE2900B666.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_2138.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/270000/E133045A-0224-E64B-B481-7D6A58D97AED.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_2139.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/40000/81DEB772-C082-0C48-9F34-7C07D66487E5.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_2648.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/40000/27A598CA-0B61-6D43-8C0D-E6E52BAC0FFA.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_2649.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/40000/7D66D74E-2D9E-B840-94FF-C106CC70B998.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_2640.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/40000/7857F2C1-08C4-F343-9B21-570A1D713B93.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_2641.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/40000/1E1F7D97-4985-8943-8F4F-5082C03E60E3.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_2642.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/40000/83E46BD2-7AAD-0A42-A673-1700D19F2676.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_2643.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/40000/BF20422C-D595-9040-8115-5FCABA5BCB9F.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_2644.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/40000/A0C31F5F-581E-2448-8F47-B607065174A3.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_2645.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/40000/0D20840D-77B7-BD4C-891B-28F859D82548.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_2646.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/40000/6E9529BA-6302-B34B-8CA0-3129F3E816E6.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_2647.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/50000/3D196FE2-2EA7-F440-BE1E-091CB86964A3.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_2002.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/50000/44C3E7BA-C8B0-DC4E-B9C7-A0E90B0CEA13.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_2003.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/50000/7102E1E9-5F30-6A48-8D62-F56062544617.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_2000.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/50000/0EBBDF52-851B-0040-9352-6E4D737EF459.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_2001.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/50000/7DC10AE7-CE69-1B48-8E2A-5DDFE94243BE.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_2006.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/50000/7D7926CE-50B6-864F-AD9A-AE3EB4DD255A.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_2007.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/50000/459E780A-2419-8F44-93E3-C72D54264E23.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_2004.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/50000/7D4F1926-C31B-594B-BFD0-DB04D6C4D1B7.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_2005.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/50000/2F6DB7BB-6E88-C947-8306-5666F802321A.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_2008.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/50000/3CFC1901-9F2F-1C46-B399-0E479BD90CC2.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_2009.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260000/0EA3EBEE-F23F-6841-9F0A-41871D079B4F.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_506.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/270000/7354370B-ADA2-564F-85B8-6249D756F5E7.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_1263.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260001/2F84548B-D03F-A045-B298-92622F7ECA76.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_666.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/270000/4ADC5A57-6CB9-9043-B109-6AF52E2297BC.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_2176.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/270000/9D7976CA-07CE-F842-92E9-82342695F7DD.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_1261.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/270000/6784D417-1678-0D4A-866B-6DF67493F518.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_2177.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/50000/7FA50B88-AF2F-114B-B61E-B38174BDFF60.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_1799.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/50000/35B35D4D-9F73-C642-9F56-F93A774F1EEA.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_1798.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260001/63F75FB6-BF76-544C-BA01-03B4876A49CF.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_719.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260001/631FA191-2BC5-9141-A488-2991769AEF35.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_718.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260001/1E68408C-6DDF-1D4F-99E8-E6FB0D1C0EE7.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_717.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/270000/D03E67DD-7447-004A-88EF-9CFDB7CC253C.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_1267.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/50000/5B3CF612-C612-3741-B2D5-D3CD514E0FCB.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_1791.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/50000/9EBD215B-8861-F14A-AD92-649FDE065216.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_1790.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260001/1E0D2C25-E7A6-2E4B-B94A-0DE226AC4862.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_713.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260001/EEBA9B18-E5A5-8E44-BB40-FB424778ADA8.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_712.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/50000/DE5114F7-F190-A04E-ADBC-54CD3B4DBE24.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_1795.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/50000/F47902CF-4D9B-2345-9B64-5111EDF868C3.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_1794.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260000/757F9905-BE5D-6F41-9A28-CEAE40B372C0.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_661.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260001/61628465-47E7-C94F-B614-C83F85E11BB0.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_716.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260000/C3519A3B-15D6-D24E-AF1F-2EBF66A22322.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_660.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/270000/3116DB91-92C5-7D4B-97D9-F8AB3934993D.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_2178.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260000/17B1AD55-638D-AA46-8193-8E5B51C20AB9.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_1069.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260000/ACEF5D8C-F267-924B-B51E-AB6064CF5EF4.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_1068.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/270000/0E308C29-55CE-6940-BC94-09E81B3CFAF3.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_2179.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260000/37D10473-FCCB-684E-8BEE-72A048A872CE.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_1061.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260000/F530C7D6-B6CE-F346-B7B9-5CB075D98619.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_1060.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260000/68D0640C-D1D0-CB40-A65E-AE030A701340.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_1063.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260000/D1F048F8-697D-2F44-B09D-FB90831B0AFD.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_1062.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260000/08686EAC-2BF0-9848-8161-F926A417C735.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_1065.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260000/98AEA907-31DF-134A-A7DF-92C419D6A5C4.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_1064.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260000/1DAFBC8B-445E-0C44-A7AD-977058E82516.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_1067.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260000/674EDFE4-62B0-C94F-AB23-238D27FB4B40.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_1066.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/270000/ADF09196-607E-5845-BA7E-DF5F72D6ED1E.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_1669.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/270000/C0AD4F00-DD0E-6249-93A6-AC039EB5CACF.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_1668.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/270000/98D47963-5B3B-0446-8A84-A33285D230E6.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_1667.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/270000/0A4EC3EB-C28C-4041-A270-3AEFCCE5B3E5.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_1666.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/270000/52DD8452-D33A-3543-985C-02F05AEEF164.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_1665.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/270000/2E1E3999-FB11-5D49-9AA2-15708053537B.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_1664.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/270000/CFC177D1-12DC-6742-A0C6-8326BE6E1A38.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_1663.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/270000/5F17D658-2245-0A4C-8E54-D52EAC9A7D47.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_1662.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/270000/3DF69B09-AEDC-CB44-BECE-16096DDE526E.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_1661.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/270000/7A734DA8-13A6-FB41-A9BB-A2DFCC88C908.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_1660.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260000/2B1ECC85-FA26-2A49-99EE-DB46209EFF2F.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_1087.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260000/AD2B791C-D3E8-9548-8F64-CBBF140679E7.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_1086.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260000/A75D5901-C32F-5C4D-800D-840E9DE52D2B.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_1085.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260000/114F3944-5FBD-0942-9977-A55A76D658E8.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_1084.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260000/7C3CEDB4-E0B0-BD49-9063-CD4219FFA354.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_1083.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260000/09413E06-E33D-4941-BB6A-B38E2D3710B0.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_1082.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260000/561CAFF3-E50E-754D-A689-9BB9886ABC0F.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_1081.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260000/7D8DDD87-37D9-8F4D-9E7F-767BA910CF04.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_1080.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260000/2DCDF5CF-6412-1744-B387-1B72BD53447C.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_599.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260000/5544D2ED-CE93-2147-8C0E-B8A4AB564B48.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_598.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260000/74DE81C5-8919-9A4E-B7B4-FA6CF41FC67F.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_1089.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260000/574161A9-147D-314A-AE1F-D3C78F9110A9.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_1088.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260001/A0CD4963-6E70-8E47-8235-7978200E588F.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_714.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260000/10D28164-3258-0749-A550-9CF71E0F0E69.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_1409.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260000/40DAA65D-CE91-9F46-A800-41BAF9F20271.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_1408.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260000/1E45F036-30CC-364F-B4B4-C6A507B8B92D.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_1403.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260000/965C5137-527D-614C-B448-DE80BEB2C90B.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_1402.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260000/C1BF7AF3-F958-7443-9698-B7E97D58AA7C.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_1401.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260000/01C71814-B84D-1D4A-8696-61E404084059.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_1400.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260000/ED16C141-3B42-994A-A13D-18F8951F18AA.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_1407.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260000/087313BD-CAC2-DA4B-B0E8-65B3EC9C100D.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_1406.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260000/280A004B-CD30-994E-A077-AD111DA495A6.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_1405.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260000/7F7435EE-DFCB-CC4F-9B22-CDFAD7F4EB5F.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_1404.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/270000/5524CDF0-89BF-5547-BD0D-6AEC72BB85E4.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_1546.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/280000/46070B0F-BA7E-E849-B4EC-3098BC45E594.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_449.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/280000/6E0A5093-9749-364E-BFF0-3799CC1EE595.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_448.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/270000/CC0699F7-808C-2941-8E1D-1E3F87E273D5.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_1339.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/270000/C742C398-ED7C-BB41-B020-6BFE337722F3.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_1338.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/270000/5748127A-7931-9547-B7E6-7CAA4FC2EB54.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_1547.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/270000/68874170-926B-C040-AAF6-5AF2D93ADFDA.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_1335.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/270000/56EFB076-1874-EE4F-8E23-9D42E39FDEA6.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_1334.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/270000/A035F0E7-394C-B644-BEEF-13E5FFD59648.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_1337.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/270000/570402E3-5074-4C40-82BD-BACA83DB4F15.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_1336.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/270000/9EC3B938-E004-C242-B110-C20B5D84D78F.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_1331.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260001/5083D172-701F-644A-8F60-12A82133F9A2.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_690.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/270000/B89005FA-0543-7C4F-AD87-8994A3448DA5.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_1333.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/270000/A72A95C2-0EDB-AD4D-99AF-D96A2AE28EC2.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_1332.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260001/9E3090F0-4417-0741-AE30-AABC5D45135A.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_691.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/270000/A1611B5E-73DF-2E4D-8F55-110BFF4D2FBF.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_1542.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260000/69DFC7F7-8F22-9D46-BA54-00A297175F14.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_1631.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/270000/A458F6BD-A368-4A48-B7E5-AD4D70305F15.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_1543.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/280000/E87284C8-8C33-C642-91EF-427C3D760CB0.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_39.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/280000/D96E4A96-7DBF-E84B-948A-9ADDBCE78934.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_38.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260000/1660A9D1-5BA4-434B-9558-C74F678286AE.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_1540.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/280000/147A5F31-F2A5-114B-92E6-E025DCD3D918.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_33.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/280000/7DA0AD2A-B5C8-8A4F-87D3-47AE05647158.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_32.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/280000/A64AD7C1-5147-B348-BC88-72F3DC04FCA9.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_31.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/280000/EAAB35E9-2CCD-4943-A82B-F454C01C05FC.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_30.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/280000/FF287CF8-1EA8-694F-AE53-F663D4CF5B9F.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_37.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/280000/6900BEF7-53CB-524D-AE6A-EC38861B4EEE.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_36.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/280000/CAD6365B-1E4F-AE41-94B8-5B7F0FEAABF6.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_35.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/280000/6F161FB4-1B55-D24A-87BE-4DC7267C05D5.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_34.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260000/6207D14F-C8C7-124C-8288-3A97E480F3EB.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_1241.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260000/C55E1F1B-85D7-134C-82C6-3CC83909B90D.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_1536.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260000/EF12F1E4-2FA9-C44C-8271-D5A4D0F3DCB2.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_1535.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260000/F774FF1F-4E29-9649-A22C-3038A2CB4D6F.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_1534.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/270000/7AF09FA4-DDE3-3F42-A608-F0CAE660B025.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_1245.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260000/B5AC06B6-E1A1-C042-BFB1-170069BAD748.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_1532.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/270000/5559A939-08D0-9E40-90F3-ABAF86238D4E.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_1247.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260000/DCDF757E-9628-8141-A7F4-ABEA8E98E86D.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_1530.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/270000/585605FC-0888-AC45-AFA6-AA67A8DA98DB.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_1249.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260001/29E958B3-0608-7C47-B7FB-1D11227FDA0B.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_648.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260000/8A3DE335-913E-A14F-8096-0F6FC23B4990.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_1539.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260000/1574E0E6-409D-BB4F-AB62-295BE8FB33F3.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_1538.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/280000/C7BF5585-22B3-B14D-B1B8-E9ABB412B1BD.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_339.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/280000/D0A9039D-601E-E14D-B289-0C5FEEF5E55E.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_338.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/280000/F399ECAC-8F9A-7542-B8C3-A944D471E261.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_335.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/280000/025BF28A-86F8-C341-BC14-66D2F633C658.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_334.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/280000/85FBB391-0338-3C4B-AA7F-0FF8995608D7.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_337.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/280000/4FB9D282-28DC-B246-9E1D-1D0E89B63CEE.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_336.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/280000/05B57253-5141-494A-951E-5CAB33068E8E.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_331.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/280000/D25947B7-43B8-BC4E-A579-C4EC489AE81D.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_330.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/280000/4EBD5FC2-42BF-1742-BB5A-30E5222A9329.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_333.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/280000/30159881-6392-1E4E-97BC-D8876C63C50E.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_332.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/270000/6AB108F3-FFBE-B44A-AF93-F9D9C30D5C0F.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_2118.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/280000/26FFBF28-FA9F-7742-9DE8-39DEF6511433.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_8.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/270000/E39FCADD-7C33-F541-A1AA-BEFB9AD00B5A.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_2119.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/270000/75B34F3E-21CD-C741-A04E-E44DA8F0BD86.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_1353.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260000/3A888368-389A-0D48-BF9C-7B8EF5AEF68C.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_1636.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/270000/90C99767-0F5F-174A-985A-5B07EBE1FE2A.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_1462.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/270000/6FBA5990-8987-9E4C-87C3-7626138AFB65.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_1352.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/270000/9F1CAC08-F36B-B043-82ED-FC6EC5493E50.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_1454.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/280000/BD049AAE-68E9-8843-A653-3F500B16EDB4.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_2345.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/280000/B0E8AB5F-370F-2340-96B2-EB49B6BD1530.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_2344.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/280000/C09BE5AE-88DF-FD49-8262-D02E2A15E981.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_2347.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/280000/46502098-8938-1647-B807-1B9ADEA80F44.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_2346.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/280000/4F0CC6B0-9745-024F-AF2F-125EC4D3893A.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_2341.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/280000/BAB3D7AB-8839-794C-821C-03FD5944B566.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_2340.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/280000/5CFC213F-C9C2-CB46-B625-8087C4614647.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_2343.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/280000/82065692-5C62-F24E-B8F7-9C2F65C3461C.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_2342.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/280000/82FDCE69-330C-D04A-A7B5-04F5AFB2EA37.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_2349.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/280000/4B062330-AC47-E441-8006-94A0D0767C02.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_2348.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/270000/EFE6287B-DE5E-4C4C-AC0F-AEA895833FE5.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_1455.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/50000/E5684602-3B0B-D145-8FCE-AE678A114AE3.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_1889.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/50000/19E1841B-680C-224C-B3AA-95DD7D6BB328.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_1888.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/50000/F6E3CC00-D5D8-CA42-A4BE-123E382400F1.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_1887.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/50000/04BA781C-9FB7-D549-B0EE-5F579D701EFE.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_1886.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/50000/F7A00F90-0EB1-2344-BA7D-60D50E78B537.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_1885.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/50000/B1D9C772-8901-304C-BEF1-8967BCFA21A1.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_1884.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/50000/4DA1BC9F-66AA-B841-9F6D-DD19AE7C68B8.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_1883.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/50000/04222B34-3D7E-9748-9C49-58D42C48FEA4.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_1882.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/50000/DE18FDF9-7655-3443-8CA7-409F64F53EE6.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_1881.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/280001/70823E97-C8D0-7B43-856D-EAC393865E06.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_1880.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/270000/DE9115DA-2C54-3A4E-AB8E-42375D903F58.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_2121.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/270000/498D60A0-FBE0-F549-A4AE-50C7896A773F.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_2120.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/270000/AD4DD729-72CD-3348-80B4-5B10B9D30DE6.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_2123.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/270000/30AEACDF-B8CE-4444-863A-F5EBBB973FD2.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_2122.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/270000/E24F452B-8D2C-3D44-96BE-846DDB2DE86F.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_2125.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/270000/CB4B0F53-1EAC-C943-8E6B-011DE17E732C.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_2124.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260000/F829301D-0F6B-4B4B-9372-2E56E09B7008.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_948.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260000/9CC7CA66-E857-EA46-B368-38FE3D7CCBE9.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_949.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260000/5430BAAA-2030-F94F-A1AF-401D7726423E.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_946.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260000/66224DDC-EDAB-344B-9059-9408D05E7009.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_947.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260000/23E0B99D-F058-2744-813E-ACAE4D572609.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_944.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/270000/FEC57CF7-1320-2B4B-8C48-B7D12C3BBD77.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_1457.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260000/CEC42A10-DA59-8A46-9DB1-1C8EB61DB84B.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_942.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260000/E76137A1-8111-6847-AF27-8F2224A36C21.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_943.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260000/120BFE24-0ECB-1649-893C-D800CC3DE988.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_940.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260000/364712D7-7D5A-714E-B0D1-0091FE3F350B.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_941.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/40000/106F2015-B9DA-2842-9095-94A5D4AED75F.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_1594.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/280001/6C4EDE7C-FD5F-D643-9633-A77A4961EBF5.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_2410.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/40000/858FE50B-BD35-6F4D-96D8-7BB325CAD373.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_2659.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/40000/4488A874-642E-8A4B-8016-B0682A66CE0A.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_2658.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260000/357BC65C-DE14-7248-B356-E9C28AC10664.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_642.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/40000/3BE55DC7-C3D4-7C45-9D87-173D8037A91B.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_2653.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/270000/C01973A9-1569-CE4B-9C67-2B1CE6496C17.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_1450.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/40000/41EF61EF-9A34-E448-90BE-E2E8282A3F12.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_2651.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/40000/A63FF7F2-D43B-1948-B300-B9305DF42910.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_2650.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/40000/92B1332C-4895-A74A-9285-5CF2BF639B49.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_2657.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/40000/F70139CB-3EF7-1046-BCB5-0CC6F40E1DF4.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_2656.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/40000/0B566FFA-9C0C-B048-BFC3-EFFC4A5B2690.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_2655.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/40000/B026DEBD-FAFA-9D4C-BD81-793929AB889B.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_2654.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/280000/94923AC1-ADE2-D140-8FD6-8410659C5ACD.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_133.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/280000/31833A60-0FC6-B543-868E-D677AB2DA7AC.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_132.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/280000/EBC0C3D1-65FD-B44D-8852-7A2BD8A48DC9.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_131.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/280000/9B9603BA-3E89-9749-B690-84023658F7CB.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_130.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/280000/F7C08726-78CD-394B-8E30-0CE5562FBC69.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_137.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/280000/C0E34AEF-D565-3340-AA1E-6188B92B5A09.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_136.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/280000/685EEBB7-8010-6244-98E8-D446A44F92BE.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_135.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/280000/DE31D769-5964-4C46-A65C-9BB2D40C496C.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_134.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/280000/9D3681D6-5588-3048-9786-6CDA0EF5ECB3.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_139.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/280000/C2FB3075-B4A3-874E-A54D-F400449CF927.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_138.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/50000/2C16D107-E65A-A54D-87C7-8B2801C18D1D.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_2019.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/50000/791C87B2-3E1D-EC41-98BA-2F2EDD657625.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_2018.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/50000/4E2D5110-8BCD-E441-A0C9-8A30ADE64B1E.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_2015.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/50000/A98E034F-0DF6-5646-B8FA-689685B6C42F.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_2014.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/50000/225B5CBD-1D3A-E743-B8BE-C5B80E6FB555.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_2017.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/50000/949A07DF-5DC9-3B42-A453-DE152C6B4040.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_2016.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/50000/573292F1-07D2-FC43-91AD-4324323891C1.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_2011.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/50000/A0411783-3F92-9942-96D0-4A12308B386B.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_2010.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/50000/998C8E75-3617-C941-B9B7-1E085EE1B395.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_2013.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/50000/C2038BCD-5614-D14D-AF3C-960DDA57C10A.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_2012.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/270000/904CF868-8765-F246-8DD8-615872205EC1.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_2112.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260001/0E7ED639-7110-4F4F-8BD5-479DD77CD200.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_708.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260001/F670654F-944D-E84C-A18C-27E50E2369D7.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_709.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260001/57E08366-CC7E-8C44-9F5E-76724B802401.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_704.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260001/0D48CDAB-804C-9D42-BE45-AD724E768138.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_705.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260001/FBFF3104-9262-8F45-BB73-56838771EB64.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_706.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260001/21604640-3AA9-3449-B87B-8635AD66100D.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_707.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260001/E1476F60-BC15-D54A-BD85-57B0A09293E2.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_700.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260001/5209D0FC-2389-CB46-B613-67FB428C3579.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_701.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260001/494CBAAC-6BE1-D347-A2A1-CA2336663FA0.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_702.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260001/57799967-0BBC-7A41-9A19-136F9A0B10F1.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_703.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/280000/50CA582D-256F-494B-93A4-01302F66EF99.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_88.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/280000/EDDDC37C-3B50-1A4F-B996-78EC4567D4FA.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_89.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/280000/EAF084F5-BE90-B74F-9E47-0959C7DF4209.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_82.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/280000/B5ACAC8A-AE9F-F340-A8EF-BCBB5F38A220.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_83.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/280000/22B38316-DC74-654D-AF8D-C6AB9EED1D2A.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_80.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/280000/9A5F1413-A67D-2F4F-8258-EE5E199184F2.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_81.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/280000/2E85DA4F-6633-7745-93B7-7A4981F259FD.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_86.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/280000/D6BF9A9B-30A9-A44C-8574-0E3576F50597.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_87.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/280000/4FD942EE-620A-6349-BB0E-202EDA7A6C22.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_84.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/280000/87B618B3-0604-6A4D-99A1-063F5973040D.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_85.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260001/3416C10D-F1FF-7E4B-BB9A-E723C0802395.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_762.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/270000/5AD86A9E-8AF3-3B48-9D4F-65835B492E42.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_1658.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/270000/1590CC0B-C12A-3247-A586-2BE82AB3B8B1.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_1659.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/270000/C44CACB2-FD33-6E41-97CD-79C23A7F514C.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_1652.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/270000/7450530C-56F6-5848-933C-C32511FF276F.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_1653.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/270000/82EA8BBB-6231-4F41-898E-DB0CF60AFC48.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_1650.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/270000/A89085D3-F48C-7C41-9026-7B55AD8B0EDA.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_1651.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/270000/A86848CB-DDF2-C144-AE72-8B90AAD8F899.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_1656.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/270000/9EF3AAB8-5D50-0F42-897E-C0A0BABE9AAF.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_1657.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/270000/7AFF8ACE-9584-8941-8B08-22C20154473F.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_1654.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/270000/FEA6B996-8944-FE4F-8294-5C320644D4A3.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_1655.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260001/E3950BA6-F0B3-AE41-AC68-8B54531D79B0.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_586.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260001/CCCA7E46-8FDA-A347-9032-5FFC6E76554A.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_587.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260001/0987498C-8614-424F-BB3A-7AE41212E8DA.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_584.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260001/37F32487-9949-4447-B38B-198770CBFC3C.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_585.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260001/7786225B-DBB1-9B47-9EB0-9727B6AB6C50.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_763.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260001/102635D3-EE9B-6C4F-BA75-5045FF68E091.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_583.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260000/723E5DE4-26C7-0443-B3B2-E47A6D21BAB8.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_580.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260000/3B5403A0-1F8A-B240-BCA5-1C23D85DD98D.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_581.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260000/359B0A62-66F8-584D-8297-6BFB51E0F7EE.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_1632.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260001/CC0EA91F-661C-CC44-9A87-60D73812B43D.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_588.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260001/76ADF8C6-6659-5842-A1EC-7907086635AA.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_589.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260000/CCEA3720-DA88-204F-AD5A-F203F2277A93.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_1633.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260000/C64D5288-C08C-F74F-8693-450603F0DE2C.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_1634.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/270000/37B7F986-C4D3-AC45-A520-20152E14467D.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_1348.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260000/47C5CEF6-DC8A-D94F-A277-0021D5647889.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_1635.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/270000/7B503BEF-A496-F74D-B866-0B07AE6260A5.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_1436.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/270000/D6D8C6E3-3357-2048-9944-6A8AA9C68722.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_1437.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/270000/71A9FDB5-EEA3-1041-92A0-F68E4A137D6F.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_1434.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/270000/62E12E17-1157-7C48-8AF0-DEC48DF059A3.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_1435.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260000/509732A8-1916-AA46-A06C-B223108B2517.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_1432.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260000/B796216E-4C4E-AF43-8C40-0FDA857BACEB.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_1433.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260000/5236A5E8-DFF3-CA43-9670-9E533F6CCA83.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_1430.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260000/740289B1-6FEE-9D4D-8CD2-1B609BCE4A28.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_1431.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/40000/D6D2BA64-8040-DC42-9D75-5429E8AE5FED.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_1380.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260000/9C371909-41F5-4A43-84E2-026819FE27F1.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_1637.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/270000/624E1629-B53A-9B47-AB54-765FAF5A9D20.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_1438.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/270000/604709C9-A6A7-E64A-9CBB-495A2989FE90.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_1439.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/270000/ED79D018-D1B0-C64C-8AD9-4B05018FD9AF.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_1260.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/270000/188AEDE7-3931-CE42-B856-D4C891887459.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_1541.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/280000/2FC44727-3F78-5D47-BB7D-9CD071A7877F.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_458.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/270000/BC8B5C97-9CC8-F24A-8060-B181336A50D7.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_1349.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/280000/7C985BD3-4543-2A4E-B16D-67255F81AEFA.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_450.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/280000/AEB8A28C-3EA4-DD45-ABD3-A96115BF5E13.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_451.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/270000/929E6549-E20F-EB4A-A3BE-2BA65C9FEA30.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_1342.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/280000/20D242B8-8FD9-0449-92CB-2BEC0334C90B.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_453.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/280000/C472CF51-13DE-7E48-8608-D194B02CAAE6.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_454.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/280000/AFC81D38-25E6-CE4C-BADB-151A2E41C6B5.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_455.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/270000/11525E18-056F-1243-A99F-772ED36C6D8F.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_1346.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/270000/954F38C9-5110-6A40-8692-28867213553F.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_1347.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260001/42D68CA7-690A-C045-8B51-EFC1508EC280.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_517.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/270000/E959AC91-5834-1A41-B631-CB4FAAB24921.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_1266.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/40000/F2019CA5-D505-8A40-AE9E-D57E2FCA3986.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_1502.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/40000/33DAA2EF-6334-3748-950F-215825C7C7EC.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_1503.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/40000/DA72F581-F1BC-C648-A825-2F0577A1A699.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_1500.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/40000/83CAC4FC-6DEB-5E44-AAC9-AE9EA5F11DB0.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_1501.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260000/8DDD80F7-4675-9045-93D1-98B805A36923.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_1506.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260000/D76D2279-341C-144C-955D-2C511B324596.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_1507.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/40000/604A018A-5283-974C-9230-DA323EA9E5CB.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_1504.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260000/9C14F419-CB3F-B34B-A0D2-4C0619542DE2.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_1505.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260000/1F95C9AD-6A7A-F745-952D-1F764F7304B4.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_1508.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260000/FD030930-3044-F746-AA86-3B07DC3875F4.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_1509.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260001/87700820-A944-8540-90BE-802BD1577BB1.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_658.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260001/42FC25DA-6FDB-A14D-A46E-591020937D61.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_659.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/50000/A3740540-43FD-DA48-A91A-3765D52A197B.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_1992.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/270000/8170F984-466A-6F4A-A67C-6B959AA71CDD.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_2127.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/40000/F0965905-105E-7043-97E8-DB189BC4EC76.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_1598.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/270000/111D81F3-7555-7A4F-A169-62EA0098EB7B.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_2126.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/270000/B87CF013-CF6D-9C43-B587-5C15D595FE54.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_2129.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/270000/74F366AE-D44C-0A4B-81D6-BA8743A249C2.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_2128.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260000/4222A5D4-B426-3D41-BA32-8F88C1017DB3.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_631.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260000/54657002-0EE7-BC4F-99BA-6561B2DF66F5.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_945.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260000/3F106CD0-ECCD-DC4C-928D-3983A41D9F72.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_1292.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/40000/87F3C880-A73D-614E-B79E-E6308B73CB14.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_1376.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/280000/88C82385-B8FD-BE4B-AF09-909E48CB8619.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_322.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/280000/24BBD548-EA45-0448-A549-9ABD9FDE2F33.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_323.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/280000/F991845A-27DF-4245-BB74-4757ABCAED1F.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_320.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/280000/C09315EE-3E3D-7043-B1C6-E9B927EAA52D.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_321.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/280000/955DA4F8-DE24-FF45-9BA4-BFDB7356376C.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_326.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/280000/EF5BE378-0A90-0D4D-B4D2-B64482906830.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_327.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/280000/144B012F-08C8-1B42-870A-F32EBD825385.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_324.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/280000/6D24802E-26F8-BD4A-843B-7516F4D8E352.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_325.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/280000/EAEF3B86-C8DE-C94F-BCCC-750D51211241.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_328.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/280000/8138CEE8-B037-2B41-BC82-7B9D79194C61.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_329.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/270000/3406D280-E2D6-7549-B8B5-B42C72316005.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_1340.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/280000/5AE89B6A-1B2C-AF47-A967-2E36626317B3.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_2330.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/280000/7DE32C49-6F98-D54D-876D-0576A36275AB.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_2331.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/280000/03CF7ED7-4B60-614E-A152-DDA6D328E884.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_2332.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/280000/417A14BA-BDAD-0247-8635-85AD0B25B311.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_2333.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/280000/654034BA-B492-FA48-8A43-2F254EB55D4D.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_2334.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/280000/D4DC2126-C2A4-E34D-8713-660C9C751D26.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_2335.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/280000/D8BDF214-4F85-9847-B6B0-3971902C91DC.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_2336.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/280000/E49D28FD-8DCB-404E-9C6B-A0BF6B46A8D0.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_2337.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/280000/9809D636-E195-0C47-A8D7-6E268915601F.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_2338.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/280000/3A783C89-3D52-7648-B68E-4413CF2581BD.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_2339.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/270000/9F444EEF-DE46-B444-908F-2F2D18C981AD.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_1341.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/50000/213815DE-F9E4-FF49-85C1-47FD80B702A0.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_1995.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/40000/FE345D64-0F23-7E4C-B392-EB5B22BDDED7.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_1592.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/280000/51DCC244-F72B-564D-96B7-F0BD68CB3E9B.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_2488.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/280000/06745A41-2DB4-3940-B6A2-C5B9CD45FC4B.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_2489.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/280000/4B0739A0-54EE-164C-9A91-68DFC9282E8D.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_2486.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/280000/0A7EAE21-B091-D44D-A98B-F990615FA786.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_2487.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/280000/EA33348C-BA0B-6E47-B29F-9279F3E6C79F.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_2484.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/280000/7505A92F-0023-EC45-A5CE-829A99B01D18.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_2485.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/280000/6C8722CF-5364-204D-8161-BEBB4AFF9D8E.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_2482.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/280000/0B23BC46-3633-C041-8B68-ED43A795464A.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_2483.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/280000/006B8112-CA62-B245-B1E2-6B0B65146A38.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_2480.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/280000/DDA0DE25-968D-C245-B026-25FDBCB9A470.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_2481.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260000/B1F9AC1C-ECD4-9743-ACAF-719BF366F50B.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_1518.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/50000/01F5929E-682B-2049-A689-8EA5D09023C4.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_2248.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/50000/7D441078-5A97-A04F-A0FF-B029BABBE751.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_2249.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/40000/6DFCE606-93E7-CF44-A4D9-22009BAEA17A.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_2652.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/50000/B08D2BC4-19EF-F244-8531-D9BFAF59F8A3.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_2244.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/50000/F661A5BC-4536-CC45-BDB5-B83446384D5D.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_2245.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/50000/59262524-E4CC-4D48-871F-6DF06694DDE1.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_2246.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/50000/B2729D23-1CF8-D04C-859C-1DE083BA0D8C.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_2247.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/50000/49900B8E-980A-3B4A-A22B-C085E2F54833.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_2240.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/50000/47950458-B27A-4D45-9178-7EF70BFEC612.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_2241.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/50000/15689B96-B714-8E41-BAD9-A0E99F942A5E.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_2242.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/50000/3C63173F-99AF-2645-AAD5-DF704EBC42D6.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_2243.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/280001/BF3831B0-DEA0-0F45-BE33-EB8D2F26F2C3.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_2464.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/280001/69E3C2A8-FD81-344B-8AFB-E754277AF702.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_2465.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/280001/A00686AF-D31D-414B-8AFE-6C9E6675BD35.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_2466.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/280001/5A3B8C00-3054-7A40-BC70-37B17EEBA1E4.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_2467.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/280001/7F9F5DA9-2EAA-BE43-8553-266E54416340.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_2460.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/280001/4ED1D996-7247-D948-9B50-549940F603BC.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_2461.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/280001/D7FC1DA3-9203-AF46-850D-5C9BD82BF29F.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_2462.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/280001/C9819AB2-4E81-3846-A187-58A0E730A1DC.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_2463.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/280001/C39490BC-43CB-BB4D-8CAD-CD6EFE04F4B5.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_2468.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/280001/96ACDB02-2B2B-524B-BBFB-68E8908A345D.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_2469.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/280000/F12FC54D-4EAB-1747-A569-8F402037BD07.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_120.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/280000/8014DA03-D42E-CE45-9AD6-83D04C5CED98.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_121.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/280000/B2403DB5-13F2-974C-8B3B-5C72611C0CEE.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_122.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260000/966B26F2-83D0-604E-95EE-068FAFA040AD.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_1026.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/280000/4D729F83-88EC-1F47-8219-0215DAF2D332.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_124.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/280000/33413670-1940-4F40-85DA-DC0D2DB58296.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_125.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/280000/2E1D2E28-B825-1846-9DC1-B6C2553D1763.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_126.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/280000/1462E690-23A9-6048-96F7-595EDEC8D3AF.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_127.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/280000/6466BECD-2AAD-BB42-A254-ADFC77A56D04.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_128.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/280000/F12C2EAF-0B4C-B944-B0E7-CCA5353FB3B7.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_129.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260000/58F7D6D1-E7A4-5D40-88AD-05AD0A955529.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_1295.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/50000/12201ACF-5E5B-8A4D-A218-3E2F105056E6.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_2068.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/50000/3633143B-2B2A-7F46-B35B-71B46008B654.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_2069.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/50000/E4085970-ACB4-B742-AD92-3DC5167B5408.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_2060.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/50000/98C58655-FC36-644B-BC0F-F0953266D538.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_2061.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/50000/546AD7BD-AB1B-644E-967E-BDA7A588101A.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_2062.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/50000/5B9B3921-78D2-3B46-8EF3-4AD18658558E.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_2063.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/50000/2FDC0CBF-94D0-364B-B1F9-3B367D1377D9.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_2064.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/50000/73CA9A84-5884-3D4E-844D-C36B61153C62.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_2065.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/50000/7AD8376B-BC48-9C44-9A20-3F1CCFBAE702.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_2066.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/50000/8AC21FCB-51A4-3646-9809-8E14402C7EE2.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_2067.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/40000/6BEF86A6-B572-EC4E-A808-815806F7DD95.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_2666.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/40000/F2F9E27D-EBE0-994C-9F4C-4C777A70D852.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_2667.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/40000/D1D59442-7AC7-D543-B5EC-CA7B849506D7.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_2664.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/40000/5B7F7D68-802F-C049-AE49-B41A65D6521D.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_2665.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/40000/0A216D71-1A6F-E147-81A2-3A32B8E27E83.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_2662.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/40000/03A4E16B-9BD3-DD42-830F-1839E819BFE0.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_2663.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/40000/B00A6AD7-2788-534F-9DBE-08E2C8B2417A.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_2660.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/40000/244EF867-634B-CA42-95FE-DE430DF6474D.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_2661.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260000/69C66762-80FC-E44D-A9AA-07A867E2C0AF.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_722.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/40000/138097EC-C4C4-9D4E-906C-724FDE30454E.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_2668.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/40000/48E26634-4CC4-C442-BC07-F936B486879A.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_2669.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260000/CB5680F5-9DE9-6C4B-997B-C5EE94690FB9.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_1645.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260000/F9263C2F-9988-C542-A3CB-453045122775.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_1644.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260000/79757017-2453-BA42-B8AB-3B5B304C7CF0.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_1647.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260000/EE94E740-5B35-8E4A-A2DF-67B94FB307FA.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_1646.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260000/96B96B72-4309-FF48-A50E-4C7EEED5B828.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_1641.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260000/20B62F76-8340-6C45-A6A6-45338F41391A.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_1640.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260000/47BDF48E-93B6-6A44-9E59-F93350B381CF.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_1643.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260000/9A3AA761-77F5-3D41-97BE-95418C6190C8.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_1642.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/270000/1FDF6CA2-89A0-634E-8221-B169BB108E11.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_1649.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/270000/0675B9EA-4565-1F45-944C-FAB048A6C069.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_1648.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260001/16237EE8-BE5B-4C4E-B0C9-FE1DBC53E968.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_728.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260001/56BC5486-B184-AE41-B398-A2E48F4CB9C3.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_729.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260000/4432024D-B27A-8F4E-9548-F50D6DF71841.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_579.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260001/B2D4BEF2-F69D-964F-85B2-0D7B77AB6F51.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_578.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260001/4CFCE39C-71EB-6B4C-9802-8A1BCA433E76.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_573.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260000/68BE1656-5945-5441-B268-2F5EF8102468.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_572.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260000/9DEB1BE1-6253-B143-86EE-94E40D9E0DA5.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_571.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260001/85809A81-487D-0849-A3BA-AD742FCBA1D8.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_570.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260001/FA3A4BBD-06C8-E146-A254-097BFDDA2F1E.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_577.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260001/C2C8CE1B-2F63-A244-91D1-583F379EC00B.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_576.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260001/E6139A59-49FF-1540-BDF9-6AA07AF01AF6.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_575.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260001/AE49F1E8-9A10-664B-A4FB-CCFCCB0E1F1B.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_574.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260000/F51977AF-F5EF-CD4C-989F-DE6569783D55.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_1209.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260000/7F0AE800-67A7-304F-B5F4-82D2049524E7.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_1208.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260000/B15553E7-8760-6644-94EF-F3170242B3C7.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_1421.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/270000/38323815-735C-5F4A-BC1A-7F7361A934FD.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_2154.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/50000/5A76F0D3-1E6C-C543-9F8F-9CF8353B3824.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_1997.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260000/5AA0CC12-7D3E-3947-A34A-4A6EE845E6DC.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_1422.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260000/94A55923-8987-2B4E-ACEA-442DAA8FB6A9.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_1425.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260000/E74DA934-E3A8-A94E-8B37-C1BFCCDDEC96.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_1424.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260000/87CD3A17-345D-A54C-9ADC-12145F4B4BA7.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_1427.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/280001/17FDA2C0-4616-CC43-AD21-6A54A1AB02F0.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_2407.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260000/61AAF626-FCFC-9A47-AB5D-202228E8A818.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_1429.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260000/AC67E883-06C0-6447-BCC5-63E8349C6191.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_1428.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260000/74EB7BD6-E7B3-1A47-8C95-FC6518A4BB92.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_939.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260001/2AC33479-C5F1-4D43-AC40-2FE073DD78C0.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_731.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260001/A83E28E8-507E-EA4C-9019-5C56D55D806E.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_730.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260001/A36CC9F8-B453-4344-8935-8FADA383DCB3.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_733.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260000/E63A1763-25A5-FE4C-84F0-E3A42A5DF317.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_938.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260001/9BFD6AA1-D46F-184B-9E04-B499626B7EE9.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_735.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260001/E9971E11-E0A8-D640-BFB7-DD78F854184D.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_734.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260001/2DA2203A-458E-2247-9607-3344E2FF10E5.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_737.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260001/D866D570-A458-1544-9B71-E7863B5423FB.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_736.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260001/AB17D151-E538-BF42-A72A-977669F6BBAC.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_739.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/280001/659210BA-F4A7-794D-871D-4165D1A6B86E.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_2402.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/280001/7BC69276-53E2-5C43-AB58-DF04F1683D5D.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_2403.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/270000/837BC960-B3C3-6149-B573-46E61ABA1F31.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_1359.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/270000/5B926FCE-4039-7943-AA02-DC00A3352D63.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_1358.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/280000/B362272A-97AD-A946-99DC-8BB79BBEAB2B.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_469.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/280001/42990F99-1744-9948-B4D3-65CED1AFC2A5.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_2400.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/280000/CE6AE6B1-C925-F149-A275-2E1DE0EC9494.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_465.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/280000/20763721-616E-804C-9DFF-C4C7F3940B8F.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_464.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/280000/15EA2727-3485-9B4A-8E6A-6B8EC8C23B89.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_467.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/280000/8C31AF70-561B-DC48-A0EB-C896DA232E6E.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_466.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/270000/1E1201E9-96E8-A842-8D8F-EA3453AD2379.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_1357.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/280000/970E9E9D-0440-F146-8EEF-92D4DCFC8F60.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_460.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/270000/4D923C89-DC62-6349-B076-BDDECDDD7614.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_1355.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260001/63417E40-D47E-C04C-8A52-14713022B03B.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_608.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/270000/840A3DCF-123A-D241-9464-ADDA87FAAD77.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_1273.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/270000/2AB61406-D0BA-2249-93A0-2FC165D62DBD.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_2158.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/270000/D0502A60-C950-6C49-9960-6152E1B7BDD0.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_2159.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260000/3275EE41-4667-AD45-B634-EC5BF8DE66E1.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_1519.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/280001/1F72CF82-6362-BE47-BAAC-5BFA0200DA9C.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_2408.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260000/2DF9F95D-35C6-454C-BEF1-4996985B9148.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_1515.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260000/F1498273-4F1D-C746-AB0A-9EFCD07FA033.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_1514.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260000/70C784BA-FF67-3445-B725-87D33C131525.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_1517.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/280001/D23A6A7D-B8F7-034B-A42E-9E39CA280540.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_2409.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260000/3D0AD026-E734-4F42-90B7-B2123B7736A2.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_1511.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260000/D49AA549-17B3-EF47-8AB4-773DC3272F9F.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_1510.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260000/CE538070-E626-B44E-9161-27B430B11D3C.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_1513.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260000/349BE61E-418D-FD4E-9220-801B3A80690C.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_1512.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/280000/070E3B4A-D2DB-8E4D-AAB9-32F5EFC219BD.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_281.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/270000/40BA17EB-B454-7241-90E0-0EAC21AFEDFA.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_1275.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260000/0DB2F6E9-9E2F-0B48-BC9A-75620901E60D.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_1004.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/280000/2B8CC8C4-D0DA-ED47-88D5-57CB3911340C.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_280.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260001/543E3B84-A778-B64C-AE30-3E8D80DA2EB8.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_723.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260000/B51E91F0-BC91-A940-BE0A-4339EE6890C6.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_582.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/280000/961BE911-D62D-2140-97F2-C4D78DDB10B9.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_357.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/280000/FC324E12-4A4F-CA4E-AB7B-EBC43BAC608A.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_356.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/280000/7B6B3FCA-ECF8-D748-AFA5-F05CAF6E1873.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_355.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/280000/07445B4B-EE39-134D-8F95-E029CE5DA781.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_354.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/280000/93D1403F-54D6-EF4C-ADD7-E6E4053070BA.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_353.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/280000/DBF1292F-7F38-3A49-9885-5B79E54919F7.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_352.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/280000/C056CAC1-9E22-6047-92CE-EDF71957AE56.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_351.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/280000/4C271803-BF4D-5142-9E26-AE15BA3B0FAC.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_350.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/280000/34D87C6F-989B-554D-AD46-32928ACB67EF.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_359.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/280000/23E3F65E-4B32-DD49-BF67-C34DD81BAF20.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_358.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/280000/3FBE7549-1927-0E40-9E25-8E02556EF3D3.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_43.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/50000/00885162-52E1-4145-B9D2-564438BE30F8.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_2323.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/50000/755A15CF-6609-EB49-9748-E3F4CDBF5442.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_2322.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/50000/F175C22F-B777-384A-B9BE-28D2E083CC07.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_2321.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/50000/DFAA9E69-0484-F340-AF33-6F72DC9F98EA.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_2320.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/280001/2B65E323-52F3-EE40-9382-FEDA15E01B96.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_2327.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/280001/BF98CC61-2C1C-894A-9B26-4616C57EDEA4.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_2326.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/280000/E98FE9F7-0807-5146-948D-A4B65A6DA71A.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_2325.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/50000/1033DA5E-7D99-F643-8ED6-D289953E529C.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_2324.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/280000/77B1AC88-F649-0441-93DC-C6682AEB6046.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_2329.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/280001/6362DD9F-21F0-634A-B1C4-A1638AF785A1.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_2328.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/270000/DAB4F6CD-9082-4D49-8CF4-D24DA3A31407.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_1445.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/280000/01C2C467-841D-B94B-A20C-F9F558C713E0.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_484.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/280000/0352F96A-FE8A-D742-99AF-40958B53BFA7.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_289.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/280000/F19678F6-6B5B-C94F-A4A9-7738CBC9E7E3.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_288.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/270000/9A2E4A96-FEBA-E949-AA67-3AF8B356B88D.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_1444.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/50000/EB613154-A6CB-5B44-B66B-6841D9691D4E.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_1793.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/280000/F16D93A5-75F9-6443-BBCF-D5890914625C.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_5.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/280000/CC3AB585-EB51-4740-A468-C057BF15A58E.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_283.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/280000/F9DC054D-8CE8-EE42-B027-D032DD29EE6E.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_282.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/280001/D8467703-5EC3-FD41-A01E-1BE165FE50BD.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_285.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/280000/D852C4A8-7669-0F44-9875-BE154556F44C.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_284.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/280000/86E31F00-982D-2B4A-8917-DA13BEEB6EB3.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_287.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/270000/B639EFC6-8678-034A-BF0B-F4086BE28DF6.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_1442.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/270000/FF1BBA69-9367-684B-8DCC-73A139525663.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_1441.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260000/1F866AF6-E74A-8F49-8B9F-D77DA39054EA.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_1116.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/280000/7BE72BDE-C744-CC47-A041-D05184DAB6FD.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_263.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/280000/96698277-D071-C842-8863-694C42436049.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_262.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/280000/68AE4B87-BD47-474A-B76A-864F6E27165F.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_261.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/280000/4C55DDF2-6DEC-C345-949E-48068AC9F4AB.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_260.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/280000/CB04CD47-9109-5C4C-AA99-FB56EB584465.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_267.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/280000/A9F371DB-B15D-C044-A253-5C0E8F0A2D23.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_266.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/50000/8FD2B7F4-C16C-5B45-A4B9-111C3FDF7AB1.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_1792.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/280000/34B2D192-EF1A-A249-946B-E432FFA78C08.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_264.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/280001/781C5BBF-97C8-844D-BD96-1727F903EFA2.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_286.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/280001/FCF1632A-BF2E-234A-9AFC-729D698A10A5.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_269.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/280001/F3C00B94-F372-6243-8C8E-0DC3AC5577A2.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_268.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/40000/8A304A2E-51DD-6D46-856D-19B4D600BEBC.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_1290.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/280000/057AE4BD-5B00-BF41-B7AE-38B8B390FD90.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_2499.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/280000/76474DF4-A1FB-394E-988C-136F96201FC9.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_2498.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260000/3FFC7575-039B-0C4C-9C9E-50D3F3ED75F3.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_1291.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/280000/62BD957A-4491-8048-80B4-25FC46B246C3.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_2491.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/280000/8D2E206B-1BF6-4145-A541-7F1BD4A22329.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_2490.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/280000/64EE7755-50E0-1E49-ACFA-64C793A5824B.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_2493.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/280000/F58C6523-FB1C-AB49-87F0-3F3FE1FB19E8.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_2492.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/280000/44BA33AE-00CB-B444-BA38-77F18DD99ECB.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_2495.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/280000/B6265415-A898-DE4A-B76F-3DC41A390DB9.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_2494.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/280000/ABF86E51-BBC4-1140-A6A1-0FC7E4CF1B1B.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_2497.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/280000/C11C4DC8-439F-F948-B07E-76EB686214AD.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_2496.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/270000/214D3275-CE19-934B-BD19-AB7CE7A8D8FD.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_2193.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260000/24709D7D-1B64-D845-B4B5-86A28AA985B3.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_1294.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/270000/4EF59900-34EF-4241-B699-9A1A7FEEEB10.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_2259.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/270000/4F6F892B-B525-B146-B842-4E1E312E662D.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_2258.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/270000/AB3CFBEF-9542-C942-8219-1162FD33E45A.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_2257.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/270000/F88D5781-C0E7-1542-ABCA-159DD32BB4D2.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_2256.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/270000/59ED5FBB-C4B4-C34E-A40E-A20E63EE566E.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_2255.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/270000/B53E49E6-5ED7-774D-AB90-C97AF410CFB2.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_2254.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/50000/8395D70B-54B0-C440-95FF-ADB505881453.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_2253.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/50000/B93ED791-E392-D74C-AC2E-BAD01657780C.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_2252.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/50000/807971C0-FD21-4C40-8A78-A69EB5DCD6CD.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_2251.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/50000/8A837D09-3ABE-5948-8C69-738CC97B80DC.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_2250.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/280000/AF23ED15-E0A4-7C4C-8B3B-F8AFA160A600.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_2477.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/280000/8E0F0A77-3F6D-C044-9F6D-A4E8C56E572A.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_2476.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/280001/A1CD9D65-C06F-1A4A-B660-56C7BA67D7E1.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_2475.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/280001/24AB1209-35FB-F34D-8881-006B28514A9D.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_2474.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/280001/C413523A-84AA-9C4D-A4BE-43376E6648AF.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_2473.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/280001/307EE4E8-760D-1D46-A432-91D947E1C929.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_2472.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/280001/67B56993-5F9F-4347-8CC6-8DDA5066B943.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_2471.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/280001/6AE88993-FE74-4247-840E-49EDF62DBDFF.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_2470.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260000/E0D7B92F-B12A-CB4B-B63E-D97EA2A4C25F.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_982.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260000/E8398719-E3DC-8743-9BC6-42FA554EE131.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_983.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260000/3F47B91C-F5CD-D44E-8CB1-B6E382D41B34.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_980.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260000/83757FF0-30CE-A54A-A6FA-075029AC1FEF.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_981.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260000/A8C09C98-0BA7-F94E-AB76-811CE212D94E.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_986.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260000/6C898B37-C8B5-1245-98BC-FF17C795A91D.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_987.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/280000/CDE3A567-7DCC-0F44-ABE2-11E98C5A7D1F.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_2479.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/280000/516AC178-CE3C-C545-9895-0133D282C018.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_2478.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/280000/DF243B73-A95D-B74A-8D5F-66C7ED8D8495.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_115.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/280001/5182D741-0990-E540-9E53-39EB857823A1.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_252.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/280000/5E0A2347-18CE-8049-AB95-9F371F3EC344.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_117.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/280000/41A18DE0-0DB7-9549-93BB-604F84386F74.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_116.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/280000/2B3B1D4A-4958-084A-87FE-066E15FB6BA1.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_111.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/280000/6EC37C48-D69D-304C-A5FC-5CF55C9BCDFE.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_110.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/280000/CAB05904-7A53-8A40-BD5A-2EA6C29C8E1B.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_113.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/270000/08152A3C-6600-9846-BC97-175C1F43A93A.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_2199.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/280000/A23F3A03-75C3-2941-BE0E-9D84B517A89F.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_119.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/280000/DA267482-B3C0-DA49-A63C-97872CB8EFFA.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_118.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/50000/BD0EE049-C6C2-1046-A555-EE144C22FE43.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_1797.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/50000/D359702E-34E6-C144-B92A-AB7DC2866774.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_2079.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/50000/2E73788D-6CD6-AD40-B0F6-0EDDB89A16C0.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_2078.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/50000/27FD737B-03AB-B241-AAF3-4541911DDDC6.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_2073.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/50000/6C9972D5-3DEE-7241-97CD-765678FEBC0C.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_2072.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/50000/1E487AFD-B67C-4F4F-8648-5DD0314D4272.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_2071.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/50000/3FE945A1-B8F6-AF45-AD03-1F08C77A4C6F.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_2070.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/50000/A3A60FCA-6FE3-8D4B-A08A-417213692218.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_2077.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/50000/A9E310A4-72ED-CF40-977F-93714B2A260F.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_2076.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/50000/B9244237-2AF9-B046-860E-1C11F1E8DD77.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_2075.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/50000/807E78BB-082E-6245-A2B2-17CF0028CB12.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_2074.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/40000/4FED3EDD-D482-214E-90CF-E96A242F8FE5.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_1388.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/50000/55C2551D-34CD-D44C-81F9-8A7A9C895C95.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_1796.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/40000/529DCF2D-F09B-3B4D-9E42-44817D29A926.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_1389.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/40000/8A8379D8-AF65-8D47-B2F1-F2A8D5FF7CC3.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_2671.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/40000/BCDCE14E-E627-2743-80CF-2B5D602D9ACD.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_2670.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/40000/7DD49E01-387E-5C4C-8CE4-1B605E04F094.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_2673.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/40000/5B0203CB-A0EB-AF41-9EDD-DF30FFCD817D.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_2672.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/40000/7315BE90-A292-164F-8AF3-7398D7D717F1.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_2675.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/40000/00C4943A-BBC7-1844-8EC8-A1DDD954FBFA.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_2674.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/40000/5811B115-7FD3-D647-BF7E-2E35BD206E60.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_2677.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/40000/5AB13379-73D5-8C48-9C61-72DC542F6ECF.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_2676.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/40000/C6B2634C-0569-2B44-ACAD-18EF122E7E4D.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_2679.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/40000/1F32D73A-D87F-B74C-925A-88E2DEBD78AB.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_2678.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/50000/C895804B-8B39-9444-B1B4-E2E5ABB1F83D.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_1986.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/50000/CBB696B0-E024-E14E-B31F-D1181B825183.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_1987.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/50000/C0300700-8855-FC4C-B79B-46D77A3EFDD0.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_1984.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/50000/8DC3D097-0AE1-8040-901C-90F7538FC996.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_1985.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/50000/F615B97E-CE57-7441-8B72-C80B9A131E03.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_1982.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/50000/B5DB83F0-4842-1E42-AE53-259650F45F5E.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_1983.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/50000/0CACFF5F-9933-E641-9A66-CAEF9EAA1E86.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_1980.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/50000/D7C96451-D807-794F-BE77-DE223837B74D.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_1981.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260000/2A69E43F-8CC6-3E45-B555-927AA6455895.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_1638.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260000/2750BD9A-79F5-334D-943F-0DD68F55C145.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_1639.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/50000/AF9105BB-AF50-7B44-B388-3EBB6BE96027.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_1988.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/50000/5B499ACC-F0B7-4348-B38D-A9BE2AABB5BD.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_1989.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/50000/DD3BF605-86D0-FD40-B1CA-08E48C2FF97E.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_1781.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260001/B57694DF-C582-AA4C-A30A-F1E17AF36967.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_568.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260001/0C4756A0-DB70-6245-BE21-2D9227779D18.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_569.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260000/3171BC3F-5156-D24A-8D84-581CEFB9108E.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_560.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260000/62B2412D-A050-D04C-8B40-293C40D4911B.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_561.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260001/E0BEFC5A-2892-FF43-859A-FD4259993AD8.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_562.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260001/E2215025-E2E5-DC42-936E-18F6094D1B0B.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_563.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260001/8049D9DF-B032-DE49-8BCD-11F1BFED949B.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_564.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260001/DA4FAD19-2AFC-084F-A1FA-E76EE6303E04.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_565.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260001/10DC2100-48CC-3148-BE68-196741E12630.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_566.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260001/B3FEB46D-C4E6-CB4B-AE11-3645726B9AC1.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_567.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/270000/D798B882-E391-DF47-87EB-7E383D3C0370.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_1188.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/270000/C64E1399-840A-5149-83BB-CFE47A2CB05D.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_1189.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/270000/1DB8F046-26FE-384D-9991-E927CE05C9E1.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_1186.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/270000/359C7B9C-14ED-2047-9C3D-63EFB9FCE444.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_1187.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/270000/B416627C-2D5C-524A-BE87-87F17ADA21D0.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_1184.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/270000/B4600A0D-B99C-4240-9F63-5A5246E65196.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_1185.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/270000/656AB422-6401-164D-8C74-89163943392F.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_1182.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/270000/5B500F22-9369-024E-B04E-32067E2B8A4E.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_1183.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/270000/6417841A-4D60-0647-873C-751D77723130.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_1180.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/270000/EAA0C7E5-AF7D-B146-8098-ED7C000517D9.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_1181.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260001/26456761-2420-D346-B17E-544B5478C731.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_726.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260001/4D528576-B562-1D45-B242-CD5C06D12406.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_727.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260001/55D2E987-B38A-0446-980E-FA4204F20C70.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_724.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260001/F8CE6012-0B91-BD4B-9D02-4ACDA0DE8EE8.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_725.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/50000/06E23950-5397-584B-B2B0-20D14F24915E.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_1748.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/50000/A9B3F804-5756-8C4E-A842-276AED4919A9.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_1749.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260001/5D56AA74-CC46-D74C-B45C-30528BAC3DF9.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_720.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260000/E6EFA5FB-5596-084B-9DE3-3DF395130C31.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_721.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/50000/550A401B-02D9-474C-ACF5-F2DDA3EB9933.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_1744.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/50000/81D73EAE-4E68-3A42-BCA9-F4A29BF321DF.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_1745.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/50000/F08703C7-824E-124D-957D-B721FC5BD2EF.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_1746.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/50000/30D05C24-755E-684A-8CFB-D73D81742A7B.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_1747.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/50000/1B699A5E-18C1-8847-A78B-416740208FEB.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_1740.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/50000/ED58EA96-0E70-7347-9691-0D49CF3FB13A.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_1741.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/50000/AEF337FC-D569-4744-8266-BF032F5C1A13.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_1742.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/50000/389D4C0A-EA91-BC44-BBDD-4DA69164A9DC.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_1743.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260000/A5642BA0-DB98-F546-B50D-E2A1A8EDB098.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_1164.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260000/5C508AAC-0930-0440-B074-628CB2D36BBC.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_1165.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260000/6A64D1FE-978A-8548-AB43-44BB21A0E040.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_1166.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260000/A9AA595D-08A9-2E42-8F28-350185FB5D5E.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_1167.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260000/E4735266-61DD-1747-A706-BDBC5C2E0184.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_1160.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260000/DBE38A9D-049E-5449-B609-A1B780A7BB19.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_1161.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260000/8D8E0620-7CD5-4D40-954F-0B7EA1674457.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_640.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260000/6104DB8B-6325-DE40-985C-3840A90F0EA1.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_1163.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260000/D4FF9A57-FC8B-3348-AD86-1790B00D69BE.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_1168.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260000/C769D2CA-1B6D-BC4E-BBDD-8457E141F672.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_1169.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/280000/7CAD2733-8625-4845-90DD-EBC0CFAA2691.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_472.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/280000/B8F1E714-E8D2-0A4A-9579-D944E33D65E6.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_48.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/280000/DE8126A6-7820-7145-97AF-6C2AE97415BE.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_49.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/280000/27B45CC0-EBC4-6542-82C1-862637F2C4A2.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_46.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/280000/E1222794-692B-0D4F-A0F3-A9A2D6DC59A1.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_47.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/280000/4D4E7681-CBDD-DA46-BF9F-84FE25F69AFF.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_44.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/280000/C57C20AA-2E2D-2243-AE66-C3DAB96ACC31.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_45.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/280000/055C253B-B020-9A4E-83C3-385C647B6DD0.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_42.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/50000/C6916EC4-6CB9-FA45-A7DD-A1F3244D85DF.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_2034.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/280000/5DD1B7D7-DBA0-9645-88AE-35E7E9806F79.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_40.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/280000/0331520F-40E9-F749-A039-C5364E06C6D3.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_41.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/270000/28A2038A-555D-644D-9CEC-60B1D0B904F9.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_1568.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/270000/70810A59-D817-154E-A5F5-45F000962830.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_1569.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260000/8EC14050-D6BE-0146-9D1B-0BF345D85A43.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_1298.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260000/06DF5368-B816-114F-B363-CBB4D6A32E28.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_1299.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/270000/698D4104-B0CD-C04A-BEC6-CCFB5DBCE910.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_1560.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/270000/D4B6446B-5B1C-6646-AF21-1911FFBB73D5.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_1561.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/270000/DA904A25-01D1-5247-A5FC-35E7C5BF7431.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_1562.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/270000/CB2A1758-8CE3-4047-AC8E-D23E233CC665.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_1563.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/270000/D1939144-A2A2-DB41-99EA-FEF0D009A5FD.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_1564.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/270000/556326C6-4CFE-6140-A130-979BE5F7A13D.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_1565.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/270000/02D8E74E-1605-284A-9B53-E38A28D3AF32.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_1566.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/270000/A19C7F64-7777-D64E-A206-D8ADF5821EE3.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_1567.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/40000/3FCADB7C-2C85-0F4B-A5EB-0E5304AB17C9.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_1713.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/270000/9E953C27-DBE5-EC4E-914E-A4D4470F829C.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_1360.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260001/DDD0DFAA-5452-B940-A58D-4D4297C04F25.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_796.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260000/626CE012-C9CF-4547-9D42-4AA3C3FB1AB2.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_591.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/270000/E14D0B5D-20C6-EA46-8891-5673BEBCEF9C.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_1361.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/40000/1205E816-3685-AD4D-846F-6C0798269902.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_1711.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/40000/2787A059-65DE-6E44-9E7E-1CD36AA255DA.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_1381.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/40000/C4B674D2-D4C0-4B47-B657-E0D286390DE4.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_1710.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/40000/09647066-CBB8-1B49-AC87-139202082065.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_1717.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/40000/7AF789BC-C571-BD49-9DBA-E00E6946DF99.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_1716.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/40000/C46B744B-7FFD-CD4A-AE55-4C6B1056965E.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_1715.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260000/C087537D-C91B-E741-8AB0-AA467DDC718E.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_590.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260000/1803F802-3613-C34C-86D7-BD4B68598AFB.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_732.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/270000/773D6DFC-83F9-EA4F-B86F-603B4BFCDE02.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_1201.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260000/6851D972-3FE5-0048-A385-4C1DE5D12799.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_790.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/40000/FFE8441D-6E2D-1A46-8B35-889314F6E0AB.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_1366.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/40000/78AAF303-BD72-A746-BF1B-B6C1F37B37AC.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_1367.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/280000/C559AE4D-608D-6442-9926-22B993E65D49.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_470.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/280000/C251A940-EF5C-3048-88B2-CA839FA8F135.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_471.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/280000/3EE41984-E60B-C84D-8413-DB0E2FFDEEBB.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_476.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/280000/4162A5A9-28C2-484B-99EA-757A038937D6.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_477.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/280000/E653F0F9-25CE-7E45-8718-9E18AE894BE5.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_474.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/280000/91A48586-1E88-F64A-BD74-D6B69E9A46A5.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_475.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/280000/D5D619A7-58E1-164B-9102-BB9CEF7F3F45.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_478.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/280000/6BAD035A-9E96-EB4F-962B-136D70475942.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_479.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/40000/552F7F30-33E3-5845-A665-CEA4367DD07F.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_1368.root -a '-o ./ -d --year UL2018A'"
queue
arguments="-f /store/data/Run2018A/JetHT/MINIAOD/UL2018_MiniAODv2-v1/40000/9CD8B7AF-CC02-B84B-8FF3-56EDC0BBE3BD.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018A-UL2018_MiniAODv2-v1/out_1369.root -a '-o ./ -d --year UL2018A'"
queue
