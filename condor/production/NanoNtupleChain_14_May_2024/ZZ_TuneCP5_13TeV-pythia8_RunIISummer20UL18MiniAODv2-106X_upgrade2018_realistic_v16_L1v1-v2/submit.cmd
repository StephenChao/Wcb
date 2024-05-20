universe=Vanilla
+ProjectName = "cms.org"
requirements = (OpSysAndVer =?= "CentOS7")
RequestMemory = 2048
RequestCpus = 1
executable=/home/legao/Wcb/condor/production/NanoNtupleChain_14_May_2024/ZZ_TuneCP5_13TeV-pythia8_RunIISummer20UL18MiniAODv2-106X_upgrade2018_realistic_v16_L1v1-v2/exe_UL18_Wcb_NanoNtupleChain.sh
transfer_executable=True
transfer_input_files=/home/legao/Wcb/condor/x509up_u14433,/home/legao/Wcb/condor/Scripts/fetchFiles.py
transfer_output_files=""
log=/home/legao/Wcb/condor/production/NanoNtupleChain_14_May_2024/ZZ_TuneCP5_13TeV-pythia8_RunIISummer20UL18MiniAODv2-106X_upgrade2018_realistic_v16_L1v1-v2/log//$(Cluster).log
output=/home/legao/Wcb/condor/production/NanoNtupleChain_14_May_2024/ZZ_TuneCP5_13TeV-pythia8_RunIISummer20UL18MiniAODv2-106X_upgrade2018_realistic_v16_L1v1-v2/log//$(Cluster).out
error=/home/legao/Wcb/condor/production/NanoNtupleChain_14_May_2024/ZZ_TuneCP5_13TeV-pythia8_RunIISummer20UL18MiniAODv2-106X_upgrade2018_realistic_v16_L1v1-v2/log//$(Cluster).err
notification=Never
should_transfer_files = YES
when_to_transfer_output = ON_EXIT
x509userproxy=x509up_u14433
+MaxRuntime=150000
on_exit_remove        = (ExitBySignal == False) && (ExitCode == 0)
on_exit_hold          = (ExitBySignal == True) || (ExitCode != 0)
on_exit_hold_reason   = strcat("Job held by ON_EXIT_HOLD due to ", ifThenElse((ExitBySignal == True), strcat("exit signal ", ExitSignal), strcat("exit code ", ExitCode)), ".")
periodic_release      = (NumJobStarts < 3) && ((CurrentTime - EnteredCurrentStatus) > 60*60)
arguments="-f /store/mc/RunIISummer20UL18MiniAODv2/ZZ_TuneCP5_13TeV-pythia8/MINIAODSIM/106X_upgrade2018_realistic_v16_L1v1-v2/2430000/206C169B-7377-AB40-8D06-31A5448B3A7D.root -o /eos/user/l/legao/NtupleStore/V0/2018/MC/ZZ_TuneCP5_13TeV-pythia8_RunIISummer20UL18MiniAODv2-106X_upgrade2018_realistic_v16_L1v1-v2/out_24.root -a '-o ./ -m --year 2018'"
queue
arguments="-f /store/mc/RunIISummer20UL18MiniAODv2/ZZ_TuneCP5_13TeV-pythia8/MINIAODSIM/106X_upgrade2018_realistic_v16_L1v1-v2/2430000/5B2123BB-BC35-1049-B1DA-668034FA3C4F.root -o /eos/user/l/legao/NtupleStore/V0/2018/MC/ZZ_TuneCP5_13TeV-pythia8_RunIISummer20UL18MiniAODv2-106X_upgrade2018_realistic_v16_L1v1-v2/out_25.root -a '-o ./ -m --year 2018'"
queue
arguments="-f /store/mc/RunIISummer20UL18MiniAODv2/ZZ_TuneCP5_13TeV-pythia8/MINIAODSIM/106X_upgrade2018_realistic_v16_L1v1-v2/2430000/F61B9046-3431-AF49-A32E-40277497B8CB.root -o /eos/user/l/legao/NtupleStore/V0/2018/MC/ZZ_TuneCP5_13TeV-pythia8_RunIISummer20UL18MiniAODv2-106X_upgrade2018_realistic_v16_L1v1-v2/out_26.root -a '-o ./ -m --year 2018'"
queue
arguments="-f /store/mc/RunIISummer20UL18MiniAODv2/ZZ_TuneCP5_13TeV-pythia8/MINIAODSIM/106X_upgrade2018_realistic_v16_L1v1-v2/2430000/418E93C9-4186-3E48-9ECD-3B61EFBF32C1.root -o /eos/user/l/legao/NtupleStore/V0/2018/MC/ZZ_TuneCP5_13TeV-pythia8_RunIISummer20UL18MiniAODv2-106X_upgrade2018_realistic_v16_L1v1-v2/out_27.root -a '-o ./ -m --year 2018'"
queue
arguments="-f /store/mc/RunIISummer20UL18MiniAODv2/ZZ_TuneCP5_13TeV-pythia8/MINIAODSIM/106X_upgrade2018_realistic_v16_L1v1-v2/130000/646C7CE7-C3CF-AB4A-A261-5B6072B0327F.root -o /eos/user/l/legao/NtupleStore/V0/2018/MC/ZZ_TuneCP5_13TeV-pythia8_RunIISummer20UL18MiniAODv2-106X_upgrade2018_realistic_v16_L1v1-v2/out_20.root -a '-o ./ -m --year 2018'"
queue
arguments="-f /store/mc/RunIISummer20UL18MiniAODv2/ZZ_TuneCP5_13TeV-pythia8/MINIAODSIM/106X_upgrade2018_realistic_v16_L1v1-v2/130000/557E96BF-F6D9-8841-A641-28C1B15CCC08.root -o /eos/user/l/legao/NtupleStore/V0/2018/MC/ZZ_TuneCP5_13TeV-pythia8_RunIISummer20UL18MiniAODv2-106X_upgrade2018_realistic_v16_L1v1-v2/out_21.root -a '-o ./ -m --year 2018'"
queue
arguments="-f /store/mc/RunIISummer20UL18MiniAODv2/ZZ_TuneCP5_13TeV-pythia8/MINIAODSIM/106X_upgrade2018_realistic_v16_L1v1-v2/130000/98FB7251-6F7F-0240-8F62-F6F8757F0B8C.root -o /eos/user/l/legao/NtupleStore/V0/2018/MC/ZZ_TuneCP5_13TeV-pythia8_RunIISummer20UL18MiniAODv2-106X_upgrade2018_realistic_v16_L1v1-v2/out_22.root -a '-o ./ -m --year 2018'"
queue
arguments="-f /store/mc/RunIISummer20UL18MiniAODv2/ZZ_TuneCP5_13TeV-pythia8/MINIAODSIM/106X_upgrade2018_realistic_v16_L1v1-v2/2430000/ED828182-53C2-854E-A25E-E21F86A76195.root -o /eos/user/l/legao/NtupleStore/V0/2018/MC/ZZ_TuneCP5_13TeV-pythia8_RunIISummer20UL18MiniAODv2-106X_upgrade2018_realistic_v16_L1v1-v2/out_23.root -a '-o ./ -m --year 2018'"
queue
arguments="-f /store/mc/RunIISummer20UL18MiniAODv2/ZZ_TuneCP5_13TeV-pythia8/MINIAODSIM/106X_upgrade2018_realistic_v16_L1v1-v2/2430000/15BA20B3-522D-194C-AE38-78D85C6E3CF0.root -o /eos/user/l/legao/NtupleStore/V0/2018/MC/ZZ_TuneCP5_13TeV-pythia8_RunIISummer20UL18MiniAODv2-106X_upgrade2018_realistic_v16_L1v1-v2/out_28.root -a '-o ./ -m --year 2018'"
queue
arguments="-f /store/mc/RunIISummer20UL18MiniAODv2/ZZ_TuneCP5_13TeV-pythia8/MINIAODSIM/106X_upgrade2018_realistic_v16_L1v1-v2/2430000/A0F466AB-CC86-4B40-9E6E-80159DB164F2.root -o /eos/user/l/legao/NtupleStore/V0/2018/MC/ZZ_TuneCP5_13TeV-pythia8_RunIISummer20UL18MiniAODv2-106X_upgrade2018_realistic_v16_L1v1-v2/out_29.root -a '-o ./ -m --year 2018'"
queue
arguments="-f /store/mc/RunIISummer20UL18MiniAODv2/ZZ_TuneCP5_13TeV-pythia8/MINIAODSIM/106X_upgrade2018_realistic_v16_L1v1-v2/2430000/05301231-A899-F940-A4C2-6CE60A3E24D8.root -o /eos/user/l/legao/NtupleStore/V0/2018/MC/ZZ_TuneCP5_13TeV-pythia8_RunIISummer20UL18MiniAODv2-106X_upgrade2018_realistic_v16_L1v1-v2/out_40.root -a '-o ./ -m --year 2018'"
queue
arguments="-f /store/mc/RunIISummer20UL18MiniAODv2/ZZ_TuneCP5_13TeV-pythia8/MINIAODSIM/106X_upgrade2018_realistic_v16_L1v1-v2/2430000/D797DE57-DE95-8B43-BC8C-050CD9D9EC45.root -o /eos/user/l/legao/NtupleStore/V0/2018/MC/ZZ_TuneCP5_13TeV-pythia8_RunIISummer20UL18MiniAODv2-106X_upgrade2018_realistic_v16_L1v1-v2/out_41.root -a '-o ./ -m --year 2018'"
queue
arguments="-f /store/mc/RunIISummer20UL18MiniAODv2/ZZ_TuneCP5_13TeV-pythia8/MINIAODSIM/106X_upgrade2018_realistic_v16_L1v1-v2/70000/BBB7C3A5-0D0B-E54C-B020-F4DD04D47167.root -o /eos/user/l/legao/NtupleStore/V0/2018/MC/ZZ_TuneCP5_13TeV-pythia8_RunIISummer20UL18MiniAODv2-106X_upgrade2018_realistic_v16_L1v1-v2/out_1.root -a '-o ./ -m --year 2018'"
queue
arguments="-f /store/mc/RunIISummer20UL18MiniAODv2/ZZ_TuneCP5_13TeV-pythia8/MINIAODSIM/106X_upgrade2018_realistic_v16_L1v1-v2/70000/AC114860-1CB6-E04D-BDC7-5EE085650CFA.root -o /eos/user/l/legao/NtupleStore/V0/2018/MC/ZZ_TuneCP5_13TeV-pythia8_RunIISummer20UL18MiniAODv2-106X_upgrade2018_realistic_v16_L1v1-v2/out_0.root -a '-o ./ -m --year 2018'"
queue
arguments="-f /store/mc/RunIISummer20UL18MiniAODv2/ZZ_TuneCP5_13TeV-pythia8/MINIAODSIM/106X_upgrade2018_realistic_v16_L1v1-v2/110000/4D667703-E7CB-2E4B-BBC5-7604EC1EF3A4.root -o /eos/user/l/legao/NtupleStore/V0/2018/MC/ZZ_TuneCP5_13TeV-pythia8_RunIISummer20UL18MiniAODv2-106X_upgrade2018_realistic_v16_L1v1-v2/out_3.root -a '-o ./ -m --year 2018'"
queue
arguments="-f /store/mc/RunIISummer20UL18MiniAODv2/ZZ_TuneCP5_13TeV-pythia8/MINIAODSIM/106X_upgrade2018_realistic_v16_L1v1-v2/110000/DB627DDD-0B8F-DB4D-B0FF-09E25B19A3BD.root -o /eos/user/l/legao/NtupleStore/V0/2018/MC/ZZ_TuneCP5_13TeV-pythia8_RunIISummer20UL18MiniAODv2-106X_upgrade2018_realistic_v16_L1v1-v2/out_2.root -a '-o ./ -m --year 2018'"
queue
arguments="-f /store/mc/RunIISummer20UL18MiniAODv2/ZZ_TuneCP5_13TeV-pythia8/MINIAODSIM/106X_upgrade2018_realistic_v16_L1v1-v2/120000/14003CC6-AAFA-AE4F-B595-042010186C41.root -o /eos/user/l/legao/NtupleStore/V0/2018/MC/ZZ_TuneCP5_13TeV-pythia8_RunIISummer20UL18MiniAODv2-106X_upgrade2018_realistic_v16_L1v1-v2/out_5.root -a '-o ./ -m --year 2018'"
queue
arguments="-f /store/mc/RunIISummer20UL18MiniAODv2/ZZ_TuneCP5_13TeV-pythia8/MINIAODSIM/106X_upgrade2018_realistic_v16_L1v1-v2/110000/48985159-DABC-F942-8C34-607C3E292BC2.root -o /eos/user/l/legao/NtupleStore/V0/2018/MC/ZZ_TuneCP5_13TeV-pythia8_RunIISummer20UL18MiniAODv2-106X_upgrade2018_realistic_v16_L1v1-v2/out_4.root -a '-o ./ -m --year 2018'"
queue
arguments="-f /store/mc/RunIISummer20UL18MiniAODv2/ZZ_TuneCP5_13TeV-pythia8/MINIAODSIM/106X_upgrade2018_realistic_v16_L1v1-v2/120000/E22D4416-6671-C04A-B94A-E82C309DEBB9.root -o /eos/user/l/legao/NtupleStore/V0/2018/MC/ZZ_TuneCP5_13TeV-pythia8_RunIISummer20UL18MiniAODv2-106X_upgrade2018_realistic_v16_L1v1-v2/out_7.root -a '-o ./ -m --year 2018'"
queue
arguments="-f /store/mc/RunIISummer20UL18MiniAODv2/ZZ_TuneCP5_13TeV-pythia8/MINIAODSIM/106X_upgrade2018_realistic_v16_L1v1-v2/120000/1FBDEBC9-8063-DD41-B96D-89C6706A8B88.root -o /eos/user/l/legao/NtupleStore/V0/2018/MC/ZZ_TuneCP5_13TeV-pythia8_RunIISummer20UL18MiniAODv2-106X_upgrade2018_realistic_v16_L1v1-v2/out_6.root -a '-o ./ -m --year 2018'"
queue
arguments="-f /store/mc/RunIISummer20UL18MiniAODv2/ZZ_TuneCP5_13TeV-pythia8/MINIAODSIM/106X_upgrade2018_realistic_v16_L1v1-v2/70000/182A1D83-E9D6-6A43-9011-4567E6D3F4C2.root -o /eos/user/l/legao/NtupleStore/V0/2018/MC/ZZ_TuneCP5_13TeV-pythia8_RunIISummer20UL18MiniAODv2-106X_upgrade2018_realistic_v16_L1v1-v2/out_9.root -a '-o ./ -m --year 2018'"
queue
arguments="-f /store/mc/RunIISummer20UL18MiniAODv2/ZZ_TuneCP5_13TeV-pythia8/MINIAODSIM/106X_upgrade2018_realistic_v16_L1v1-v2/120000/66D47641-3019-F843-B205-4BCF64930604.root -o /eos/user/l/legao/NtupleStore/V0/2018/MC/ZZ_TuneCP5_13TeV-pythia8_RunIISummer20UL18MiniAODv2-106X_upgrade2018_realistic_v16_L1v1-v2/out_8.root -a '-o ./ -m --year 2018'"
queue
arguments="-f /store/mc/RunIISummer20UL18MiniAODv2/ZZ_TuneCP5_13TeV-pythia8/MINIAODSIM/106X_upgrade2018_realistic_v16_L1v1-v2/260000/F78AC24B-0D3F-0146-84DC-93C9180B5EA5.root -o /eos/user/l/legao/NtupleStore/V0/2018/MC/ZZ_TuneCP5_13TeV-pythia8_RunIISummer20UL18MiniAODv2-106X_upgrade2018_realistic_v16_L1v1-v2/out_13.root -a '-o ./ -m --year 2018'"
queue
arguments="-f /store/mc/RunIISummer20UL18MiniAODv2/ZZ_TuneCP5_13TeV-pythia8/MINIAODSIM/106X_upgrade2018_realistic_v16_L1v1-v2/260000/49F2AA58-1F49-0147-8A7E-9A9306841629.root -o /eos/user/l/legao/NtupleStore/V0/2018/MC/ZZ_TuneCP5_13TeV-pythia8_RunIISummer20UL18MiniAODv2-106X_upgrade2018_realistic_v16_L1v1-v2/out_12.root -a '-o ./ -m --year 2018'"
queue
arguments="-f /store/mc/RunIISummer20UL18MiniAODv2/ZZ_TuneCP5_13TeV-pythia8/MINIAODSIM/106X_upgrade2018_realistic_v16_L1v1-v2/260000/B310D1C6-B7FF-FF4D-80D7-9DD0F80761CE.root -o /eos/user/l/legao/NtupleStore/V0/2018/MC/ZZ_TuneCP5_13TeV-pythia8_RunIISummer20UL18MiniAODv2-106X_upgrade2018_realistic_v16_L1v1-v2/out_11.root -a '-o ./ -m --year 2018'"
queue
arguments="-f /store/mc/RunIISummer20UL18MiniAODv2/ZZ_TuneCP5_13TeV-pythia8/MINIAODSIM/106X_upgrade2018_realistic_v16_L1v1-v2/260000/EEFE3226-B7CE-9645-9FA9-AA2153F4AE2C.root -o /eos/user/l/legao/NtupleStore/V0/2018/MC/ZZ_TuneCP5_13TeV-pythia8_RunIISummer20UL18MiniAODv2-106X_upgrade2018_realistic_v16_L1v1-v2/out_10.root -a '-o ./ -m --year 2018'"
queue
arguments="-f /store/mc/RunIISummer20UL18MiniAODv2/ZZ_TuneCP5_13TeV-pythia8/MINIAODSIM/106X_upgrade2018_realistic_v16_L1v1-v2/2430000/68D480E6-1154-6C44-9D86-A9F755128337.root -o /eos/user/l/legao/NtupleStore/V0/2018/MC/ZZ_TuneCP5_13TeV-pythia8_RunIISummer20UL18MiniAODv2-106X_upgrade2018_realistic_v16_L1v1-v2/out_39.root -a '-o ./ -m --year 2018'"
queue
arguments="-f /store/mc/RunIISummer20UL18MiniAODv2/ZZ_TuneCP5_13TeV-pythia8/MINIAODSIM/106X_upgrade2018_realistic_v16_L1v1-v2/2430000/64EB5171-0D6E-4A48-80C8-DBE29B79B6B5.root -o /eos/user/l/legao/NtupleStore/V0/2018/MC/ZZ_TuneCP5_13TeV-pythia8_RunIISummer20UL18MiniAODv2-106X_upgrade2018_realistic_v16_L1v1-v2/out_38.root -a '-o ./ -m --year 2018'"
queue
arguments="-f /store/mc/RunIISummer20UL18MiniAODv2/ZZ_TuneCP5_13TeV-pythia8/MINIAODSIM/106X_upgrade2018_realistic_v16_L1v1-v2/260000/F799B51C-0012-A04F-BEFE-9FFF7A05865E.root -o /eos/user/l/legao/NtupleStore/V0/2018/MC/ZZ_TuneCP5_13TeV-pythia8_RunIISummer20UL18MiniAODv2-106X_upgrade2018_realistic_v16_L1v1-v2/out_15.root -a '-o ./ -m --year 2018'"
queue
arguments="-f /store/mc/RunIISummer20UL18MiniAODv2/ZZ_TuneCP5_13TeV-pythia8/MINIAODSIM/106X_upgrade2018_realistic_v16_L1v1-v2/260000/51B59A32-C5AA-2044-8A12-06FDACB0125E.root -o /eos/user/l/legao/NtupleStore/V0/2018/MC/ZZ_TuneCP5_13TeV-pythia8_RunIISummer20UL18MiniAODv2-106X_upgrade2018_realistic_v16_L1v1-v2/out_14.root -a '-o ./ -m --year 2018'"
queue
arguments="-f /store/mc/RunIISummer20UL18MiniAODv2/ZZ_TuneCP5_13TeV-pythia8/MINIAODSIM/106X_upgrade2018_realistic_v16_L1v1-v2/120000/0B0182FA-4066-8042-9AD0-A38CBDAE04F2.root -o /eos/user/l/legao/NtupleStore/V0/2018/MC/ZZ_TuneCP5_13TeV-pythia8_RunIISummer20UL18MiniAODv2-106X_upgrade2018_realistic_v16_L1v1-v2/out_17.root -a '-o ./ -m --year 2018'"
queue
arguments="-f /store/mc/RunIISummer20UL18MiniAODv2/ZZ_TuneCP5_13TeV-pythia8/MINIAODSIM/106X_upgrade2018_realistic_v16_L1v1-v2/120000/67DC050F-1659-6945-87E3-3155589AACF7.root -o /eos/user/l/legao/NtupleStore/V0/2018/MC/ZZ_TuneCP5_13TeV-pythia8_RunIISummer20UL18MiniAODv2-106X_upgrade2018_realistic_v16_L1v1-v2/out_16.root -a '-o ./ -m --year 2018'"
queue
arguments="-f /store/mc/RunIISummer20UL18MiniAODv2/ZZ_TuneCP5_13TeV-pythia8/MINIAODSIM/106X_upgrade2018_realistic_v16_L1v1-v2/280000/12FFF96E-9715-A044-B394-2627C176413C.root -o /eos/user/l/legao/NtupleStore/V0/2018/MC/ZZ_TuneCP5_13TeV-pythia8_RunIISummer20UL18MiniAODv2-106X_upgrade2018_realistic_v16_L1v1-v2/out_19.root -a '-o ./ -m --year 2018'"
queue
arguments="-f /store/mc/RunIISummer20UL18MiniAODv2/ZZ_TuneCP5_13TeV-pythia8/MINIAODSIM/106X_upgrade2018_realistic_v16_L1v1-v2/120000/3949FA66-03BF-1C43-8336-8A0A4D1E2DEF.root -o /eos/user/l/legao/NtupleStore/V0/2018/MC/ZZ_TuneCP5_13TeV-pythia8_RunIISummer20UL18MiniAODv2-106X_upgrade2018_realistic_v16_L1v1-v2/out_18.root -a '-o ./ -m --year 2018'"
queue
arguments="-f /store/mc/RunIISummer20UL18MiniAODv2/ZZ_TuneCP5_13TeV-pythia8/MINIAODSIM/106X_upgrade2018_realistic_v16_L1v1-v2/2430000/973BF7E9-9C43-8C44-B966-E1424D5FD393.root -o /eos/user/l/legao/NtupleStore/V0/2018/MC/ZZ_TuneCP5_13TeV-pythia8_RunIISummer20UL18MiniAODv2-106X_upgrade2018_realistic_v16_L1v1-v2/out_31.root -a '-o ./ -m --year 2018'"
queue
arguments="-f /store/mc/RunIISummer20UL18MiniAODv2/ZZ_TuneCP5_13TeV-pythia8/MINIAODSIM/106X_upgrade2018_realistic_v16_L1v1-v2/2430000/933A7229-920F-594C-8CE5-58FCA6DF0669.root -o /eos/user/l/legao/NtupleStore/V0/2018/MC/ZZ_TuneCP5_13TeV-pythia8_RunIISummer20UL18MiniAODv2-106X_upgrade2018_realistic_v16_L1v1-v2/out_30.root -a '-o ./ -m --year 2018'"
queue
arguments="-f /store/mc/RunIISummer20UL18MiniAODv2/ZZ_TuneCP5_13TeV-pythia8/MINIAODSIM/106X_upgrade2018_realistic_v16_L1v1-v2/2430000/08A3D060-DD31-F945-906E-70EC4EABD307.root -o /eos/user/l/legao/NtupleStore/V0/2018/MC/ZZ_TuneCP5_13TeV-pythia8_RunIISummer20UL18MiniAODv2-106X_upgrade2018_realistic_v16_L1v1-v2/out_37.root -a '-o ./ -m --year 2018'"
queue
arguments="-f /store/mc/RunIISummer20UL18MiniAODv2/ZZ_TuneCP5_13TeV-pythia8/MINIAODSIM/106X_upgrade2018_realistic_v16_L1v1-v2/2430000/B1CF949A-43A0-E649-A17B-49B74B1ADF9B.root -o /eos/user/l/legao/NtupleStore/V0/2018/MC/ZZ_TuneCP5_13TeV-pythia8_RunIISummer20UL18MiniAODv2-106X_upgrade2018_realistic_v16_L1v1-v2/out_36.root -a '-o ./ -m --year 2018'"
queue
arguments="-f /store/mc/RunIISummer20UL18MiniAODv2/ZZ_TuneCP5_13TeV-pythia8/MINIAODSIM/106X_upgrade2018_realistic_v16_L1v1-v2/2430000/05345828-6CF7-2C41-AA72-4F3A77BC615E.root -o /eos/user/l/legao/NtupleStore/V0/2018/MC/ZZ_TuneCP5_13TeV-pythia8_RunIISummer20UL18MiniAODv2-106X_upgrade2018_realistic_v16_L1v1-v2/out_35.root -a '-o ./ -m --year 2018'"
queue
arguments="-f /store/mc/RunIISummer20UL18MiniAODv2/ZZ_TuneCP5_13TeV-pythia8/MINIAODSIM/106X_upgrade2018_realistic_v16_L1v1-v2/2430000/C76F3001-2368-474E-93BD-5976A13CB18E.root -o /eos/user/l/legao/NtupleStore/V0/2018/MC/ZZ_TuneCP5_13TeV-pythia8_RunIISummer20UL18MiniAODv2-106X_upgrade2018_realistic_v16_L1v1-v2/out_34.root -a '-o ./ -m --year 2018'"
queue
arguments="-f /store/mc/RunIISummer20UL18MiniAODv2/ZZ_TuneCP5_13TeV-pythia8/MINIAODSIM/106X_upgrade2018_realistic_v16_L1v1-v2/2430000/98FA9230-6EC8-C84C-A133-0B4ABFD1BF1B.root -o /eos/user/l/legao/NtupleStore/V0/2018/MC/ZZ_TuneCP5_13TeV-pythia8_RunIISummer20UL18MiniAODv2-106X_upgrade2018_realistic_v16_L1v1-v2/out_33.root -a '-o ./ -m --year 2018'"
queue
arguments="-f /store/mc/RunIISummer20UL18MiniAODv2/ZZ_TuneCP5_13TeV-pythia8/MINIAODSIM/106X_upgrade2018_realistic_v16_L1v1-v2/2430000/E1438CA7-0719-E343-A07E-89F6B2A609F8.root -o /eos/user/l/legao/NtupleStore/V0/2018/MC/ZZ_TuneCP5_13TeV-pythia8_RunIISummer20UL18MiniAODv2-106X_upgrade2018_realistic_v16_L1v1-v2/out_32.root -a '-o ./ -m --year 2018'"
queue
