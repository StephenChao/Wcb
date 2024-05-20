
universe=Vanilla
RequestMemory = 2048
RequestCpus = 1
executable=/ospool/cms-user/yuzhe/Wcb/Wcb/condor/production/NanoNtupleChain_12_May_2024/WZ_TuneCP5_13TeV-pythia8_RunIISummer20UL17MiniAODv2-106X_mc2017_realistic_v9-v1/exe_UL17_Wcb_NanoNtupleChain.sh
transfer_executable=True
transfer_input_files=/ospool/cms-user/yuzhe/Wcb/Wcb/condor/x509up_u12976,/ospool/cms-user/yuzhe/Wcb/Wcb/condor/Scripts/fetchFiles.py
transfer_output_files=out_1.root
log=/ospool/cms-user/yuzhe/Wcb/Wcb/condor/production/NanoNtupleChain_12_May_2024/WZ_TuneCP5_13TeV-pythia8_RunIISummer20UL17MiniAODv2-106X_mc2017_realistic_v9-v1/log//$(Cluster).log
output=/ospool/cms-user/yuzhe/Wcb/Wcb/condor/production/NanoNtupleChain_12_May_2024/WZ_TuneCP5_13TeV-pythia8_RunIISummer20UL17MiniAODv2-106X_mc2017_realistic_v9-v1/log/std_logs/$(Cluster).$(Process).out
error=/ospool/cms-user/yuzhe/Wcb/Wcb/condor/production/NanoNtupleChain_12_May_2024/WZ_TuneCP5_13TeV-pythia8_RunIISummer20UL17MiniAODv2-106X_mc2017_realistic_v9-v1/log/std_logs/$(Cluster).$(Process).err
notification=Never
should_transfer_files = YES
when_to_transfer_output = ON_EXIT
x509userproxy=x509up_u12976
+MaxRuntime=150000
        
arguments="-f /store/mc/RunIISummer20UL17MiniAODv2/WZ_TuneCP5_13TeV-pythia8/MINIAODSIM/106X_mc2017_realistic_v9-v1/280000/5BEF2D36-A082-1246-B977-1B49F78A6A3B.root -a '-o ./ -m --year 2017'"
transfer_output_remaps="out_1.root = /ospool/cms-user/yuzhe/NtupleStore/Wcb/2017/MC/WZ_TuneCP5_13TeV-pythia8_RunIISummer20UL17MiniAODv2-106X_mc2017_realistic_v9-v1/out_24.root "
queue
        
arguments="-f /store/mc/RunIISummer20UL17MiniAODv2/WZ_TuneCP5_13TeV-pythia8/MINIAODSIM/106X_mc2017_realistic_v9-v1/280000/2E765558-BCEC-FB4E-99CF-C0DFFA066C79.root -a '-o ./ -m --year 2017'"
transfer_output_remaps="out_1.root = /ospool/cms-user/yuzhe/NtupleStore/Wcb/2017/MC/WZ_TuneCP5_13TeV-pythia8_RunIISummer20UL17MiniAODv2-106X_mc2017_realistic_v9-v1/out_25.root "
queue
        
arguments="-f /store/mc/RunIISummer20UL17MiniAODv2/WZ_TuneCP5_13TeV-pythia8/MINIAODSIM/106X_mc2017_realistic_v9-v1/280000/93EE820F-6C8B-1B4A-91B4-5EE11C73F554.root -a '-o ./ -m --year 2017'"
transfer_output_remaps="out_1.root = /ospool/cms-user/yuzhe/NtupleStore/Wcb/2017/MC/WZ_TuneCP5_13TeV-pythia8_RunIISummer20UL17MiniAODv2-106X_mc2017_realistic_v9-v1/out_26.root "
queue
        
arguments="-f /store/mc/RunIISummer20UL17MiniAODv2/WZ_TuneCP5_13TeV-pythia8/MINIAODSIM/106X_mc2017_realistic_v9-v1/280000/A37044E5-B951-CD4E-B26C-AB587491BAC8.root -a '-o ./ -m --year 2017'"
transfer_output_remaps="out_1.root = /ospool/cms-user/yuzhe/NtupleStore/Wcb/2017/MC/WZ_TuneCP5_13TeV-pythia8_RunIISummer20UL17MiniAODv2-106X_mc2017_realistic_v9-v1/out_27.root "
queue
        
arguments="-f /store/mc/RunIISummer20UL17MiniAODv2/WZ_TuneCP5_13TeV-pythia8/MINIAODSIM/106X_mc2017_realistic_v9-v1/280000/BF9E79C7-74D3-1947-9111-4B1D3F955F21.root -a '-o ./ -m --year 2017'"
transfer_output_remaps="out_1.root = /ospool/cms-user/yuzhe/NtupleStore/Wcb/2017/MC/WZ_TuneCP5_13TeV-pythia8_RunIISummer20UL17MiniAODv2-106X_mc2017_realistic_v9-v1/out_20.root "
queue
        
arguments="-f /store/mc/RunIISummer20UL17MiniAODv2/WZ_TuneCP5_13TeV-pythia8/MINIAODSIM/106X_mc2017_realistic_v9-v1/280000/F66572B9-FE20-6E42-B753-00525F06FCDA.root -a '-o ./ -m --year 2017'"
transfer_output_remaps="out_1.root = /ospool/cms-user/yuzhe/NtupleStore/Wcb/2017/MC/WZ_TuneCP5_13TeV-pythia8_RunIISummer20UL17MiniAODv2-106X_mc2017_realistic_v9-v1/out_21.root "
queue
        
arguments="-f /store/mc/RunIISummer20UL17MiniAODv2/WZ_TuneCP5_13TeV-pythia8/MINIAODSIM/106X_mc2017_realistic_v9-v1/280000/3086D486-9BF8-C74B-8136-81A7AE635100.root -a '-o ./ -m --year 2017'"
transfer_output_remaps="out_1.root = /ospool/cms-user/yuzhe/NtupleStore/Wcb/2017/MC/WZ_TuneCP5_13TeV-pythia8_RunIISummer20UL17MiniAODv2-106X_mc2017_realistic_v9-v1/out_22.root "
queue
        
arguments="-f /store/mc/RunIISummer20UL17MiniAODv2/WZ_TuneCP5_13TeV-pythia8/MINIAODSIM/106X_mc2017_realistic_v9-v1/280000/2B2DF8D3-E642-8541-806A-358FC0F6C8F9.root -a '-o ./ -m --year 2017'"
transfer_output_remaps="out_1.root = /ospool/cms-user/yuzhe/NtupleStore/Wcb/2017/MC/WZ_TuneCP5_13TeV-pythia8_RunIISummer20UL17MiniAODv2-106X_mc2017_realistic_v9-v1/out_23.root "
queue
        
arguments="-f /store/mc/RunIISummer20UL17MiniAODv2/WZ_TuneCP5_13TeV-pythia8/MINIAODSIM/106X_mc2017_realistic_v9-v1/280000/4AA46174-6946-9F46-8DE0-7465F018E08A.root -a '-o ./ -m --year 2017'"
transfer_output_remaps="out_1.root = /ospool/cms-user/yuzhe/NtupleStore/Wcb/2017/MC/WZ_TuneCP5_13TeV-pythia8_RunIISummer20UL17MiniAODv2-106X_mc2017_realistic_v9-v1/out_28.root "
queue
        
arguments="-f /store/mc/RunIISummer20UL17MiniAODv2/WZ_TuneCP5_13TeV-pythia8/MINIAODSIM/106X_mc2017_realistic_v9-v1/280000/331F7135-0FA3-7D4A-AC50-81380AB97EF7.root -a '-o ./ -m --year 2017'"
transfer_output_remaps="out_1.root = /ospool/cms-user/yuzhe/NtupleStore/Wcb/2017/MC/WZ_TuneCP5_13TeV-pythia8_RunIISummer20UL17MiniAODv2-106X_mc2017_realistic_v9-v1/out_29.root "
queue
        
arguments="-f /store/mc/RunIISummer20UL17MiniAODv2/WZ_TuneCP5_13TeV-pythia8/MINIAODSIM/106X_mc2017_realistic_v9-v1/50000/D0CDEDE5-F205-974C-8234-392E0D5C6C3E.root -a '-o ./ -m --year 2017'"
transfer_output_remaps="out_1.root = /ospool/cms-user/yuzhe/NtupleStore/Wcb/2017/MC/WZ_TuneCP5_13TeV-pythia8_RunIISummer20UL17MiniAODv2-106X_mc2017_realistic_v9-v1/out_0.root "
queue
        
arguments="-f /store/mc/RunIISummer20UL17MiniAODv2/WZ_TuneCP5_13TeV-pythia8/MINIAODSIM/106X_mc2017_realistic_v9-v1/50000/C70CD699-E1CC-1C4E-A21B-A38CA7ECF911.root -a '-o ./ -m --year 2017'"
transfer_output_remaps="out_1.root = /ospool/cms-user/yuzhe/NtupleStore/Wcb/2017/MC/WZ_TuneCP5_13TeV-pythia8_RunIISummer20UL17MiniAODv2-106X_mc2017_realistic_v9-v1/out_4.root "
queue
        
arguments="-f /store/mc/RunIISummer20UL17MiniAODv2/WZ_TuneCP5_13TeV-pythia8/MINIAODSIM/106X_mc2017_realistic_v9-v1/280000/C1BB1FC5-92C5-AB40-A5A9-61A3483EAE73.root -a '-o ./ -m --year 2017'"
transfer_output_remaps="out_1.root = /ospool/cms-user/yuzhe/NtupleStore/Wcb/2017/MC/WZ_TuneCP5_13TeV-pythia8_RunIISummer20UL17MiniAODv2-106X_mc2017_realistic_v9-v1/out_8.root "
queue
        
arguments="-f /store/mc/RunIISummer20UL17MiniAODv2/WZ_TuneCP5_13TeV-pythia8/MINIAODSIM/106X_mc2017_realistic_v9-v1/280000/66F53AA5-37E1-5B4C-B93F-125CE71B4CE4.root -a '-o ./ -m --year 2017'"
transfer_output_remaps="out_1.root = /ospool/cms-user/yuzhe/NtupleStore/Wcb/2017/MC/WZ_TuneCP5_13TeV-pythia8_RunIISummer20UL17MiniAODv2-106X_mc2017_realistic_v9-v1/out_59.root "
queue
        
arguments="-f /store/mc/RunIISummer20UL17MiniAODv2/WZ_TuneCP5_13TeV-pythia8/MINIAODSIM/106X_mc2017_realistic_v9-v1/280000/F2F2B944-6EDA-BD44-BBF0-089E781604D6.root -a '-o ./ -m --year 2017'"
transfer_output_remaps="out_1.root = /ospool/cms-user/yuzhe/NtupleStore/Wcb/2017/MC/WZ_TuneCP5_13TeV-pythia8_RunIISummer20UL17MiniAODv2-106X_mc2017_realistic_v9-v1/out_58.root "
queue
        
arguments="-f /store/mc/RunIISummer20UL17MiniAODv2/WZ_TuneCP5_13TeV-pythia8/MINIAODSIM/106X_mc2017_realistic_v9-v1/280000/48262055-CEC0-EF41-8CFB-4EDC74CBCAB9.root -a '-o ./ -m --year 2017'"
transfer_output_remaps="out_1.root = /ospool/cms-user/yuzhe/NtupleStore/Wcb/2017/MC/WZ_TuneCP5_13TeV-pythia8_RunIISummer20UL17MiniAODv2-106X_mc2017_realistic_v9-v1/out_55.root "
queue
        
arguments="-f /store/mc/RunIISummer20UL17MiniAODv2/WZ_TuneCP5_13TeV-pythia8/MINIAODSIM/106X_mc2017_realistic_v9-v1/280000/586BBB17-7C07-2345-80A2-4CDC60CC96B7.root -a '-o ./ -m --year 2017'"
transfer_output_remaps="out_1.root = /ospool/cms-user/yuzhe/NtupleStore/Wcb/2017/MC/WZ_TuneCP5_13TeV-pythia8_RunIISummer20UL17MiniAODv2-106X_mc2017_realistic_v9-v1/out_54.root "
queue
        
arguments="-f /store/mc/RunIISummer20UL17MiniAODv2/WZ_TuneCP5_13TeV-pythia8/MINIAODSIM/106X_mc2017_realistic_v9-v1/280000/4E62FA3D-CDEF-764F-B6A1-22D596A8422C.root -a '-o ./ -m --year 2017'"
transfer_output_remaps="out_1.root = /ospool/cms-user/yuzhe/NtupleStore/Wcb/2017/MC/WZ_TuneCP5_13TeV-pythia8_RunIISummer20UL17MiniAODv2-106X_mc2017_realistic_v9-v1/out_57.root "
queue
        
arguments="-f /store/mc/RunIISummer20UL17MiniAODv2/WZ_TuneCP5_13TeV-pythia8/MINIAODSIM/106X_mc2017_realistic_v9-v1/280000/81E829FA-E23D-A045-B82A-568303E43772.root -a '-o ./ -m --year 2017'"
transfer_output_remaps="out_1.root = /ospool/cms-user/yuzhe/NtupleStore/Wcb/2017/MC/WZ_TuneCP5_13TeV-pythia8_RunIISummer20UL17MiniAODv2-106X_mc2017_realistic_v9-v1/out_56.root "
queue
        
arguments="-f /store/mc/RunIISummer20UL17MiniAODv2/WZ_TuneCP5_13TeV-pythia8/MINIAODSIM/106X_mc2017_realistic_v9-v1/280000/8E0FE871-1C6A-0C47-9481-BF2116460642.root -a '-o ./ -m --year 2017'"
transfer_output_remaps="out_1.root = /ospool/cms-user/yuzhe/NtupleStore/Wcb/2017/MC/WZ_TuneCP5_13TeV-pythia8_RunIISummer20UL17MiniAODv2-106X_mc2017_realistic_v9-v1/out_51.root "
queue
        
arguments="-f /store/mc/RunIISummer20UL17MiniAODv2/WZ_TuneCP5_13TeV-pythia8/MINIAODSIM/106X_mc2017_realistic_v9-v1/280000/A050CBA1-FCAF-6340-8EDD-F799B138271C.root -a '-o ./ -m --year 2017'"
transfer_output_remaps="out_1.root = /ospool/cms-user/yuzhe/NtupleStore/Wcb/2017/MC/WZ_TuneCP5_13TeV-pythia8_RunIISummer20UL17MiniAODv2-106X_mc2017_realistic_v9-v1/out_50.root "
queue
        
arguments="-f /store/mc/RunIISummer20UL17MiniAODv2/WZ_TuneCP5_13TeV-pythia8/MINIAODSIM/106X_mc2017_realistic_v9-v1/280000/835C996D-CA87-2842-B6BB-7BF6F071B1A0.root -a '-o ./ -m --year 2017'"
transfer_output_remaps="out_1.root = /ospool/cms-user/yuzhe/NtupleStore/Wcb/2017/MC/WZ_TuneCP5_13TeV-pythia8_RunIISummer20UL17MiniAODv2-106X_mc2017_realistic_v9-v1/out_53.root "
queue
        
arguments="-f /store/mc/RunIISummer20UL17MiniAODv2/WZ_TuneCP5_13TeV-pythia8/MINIAODSIM/106X_mc2017_realistic_v9-v1/280000/3FC5C703-13D8-3244-B78A-92D0E682FB0E.root -a '-o ./ -m --year 2017'"
transfer_output_remaps="out_1.root = /ospool/cms-user/yuzhe/NtupleStore/Wcb/2017/MC/WZ_TuneCP5_13TeV-pythia8_RunIISummer20UL17MiniAODv2-106X_mc2017_realistic_v9-v1/out_52.root "
queue
        
arguments="-f /store/mc/RunIISummer20UL17MiniAODv2/WZ_TuneCP5_13TeV-pythia8/MINIAODSIM/106X_mc2017_realistic_v9-v1/40000/60C5E9CC-CC83-CD4B-8D00-CAC31F79ED75.root -a '-o ./ -m --year 2017'"
transfer_output_remaps="out_1.root = /ospool/cms-user/yuzhe/NtupleStore/Wcb/2017/MC/WZ_TuneCP5_13TeV-pythia8_RunIISummer20UL17MiniAODv2-106X_mc2017_realistic_v9-v1/out_88.root "
queue
        
arguments="-f /store/mc/RunIISummer20UL17MiniAODv2/WZ_TuneCP5_13TeV-pythia8/MINIAODSIM/106X_mc2017_realistic_v9-v1/40000/1C31224E-06F6-7D41-AA6C-EA98FFC2C911.root -a '-o ./ -m --year 2017'"
transfer_output_remaps="out_1.root = /ospool/cms-user/yuzhe/NtupleStore/Wcb/2017/MC/WZ_TuneCP5_13TeV-pythia8_RunIISummer20UL17MiniAODv2-106X_mc2017_realistic_v9-v1/out_89.root "
queue
        
arguments="-f /store/mc/RunIISummer20UL17MiniAODv2/WZ_TuneCP5_13TeV-pythia8/MINIAODSIM/106X_mc2017_realistic_v9-v1/40000/53723C40-6CCA-A146-A826-F1BA609E5204.root -a '-o ./ -m --year 2017'"
transfer_output_remaps="out_1.root = /ospool/cms-user/yuzhe/NtupleStore/Wcb/2017/MC/WZ_TuneCP5_13TeV-pythia8_RunIISummer20UL17MiniAODv2-106X_mc2017_realistic_v9-v1/out_82.root "
queue
        
arguments="-f /store/mc/RunIISummer20UL17MiniAODv2/WZ_TuneCP5_13TeV-pythia8/MINIAODSIM/106X_mc2017_realistic_v9-v1/40000/DBFF502B-A47E-A442-A644-5A81B1BAFB77.root -a '-o ./ -m --year 2017'"
transfer_output_remaps="out_1.root = /ospool/cms-user/yuzhe/NtupleStore/Wcb/2017/MC/WZ_TuneCP5_13TeV-pythia8_RunIISummer20UL17MiniAODv2-106X_mc2017_realistic_v9-v1/out_83.root "
queue
        
arguments="-f /store/mc/RunIISummer20UL17MiniAODv2/WZ_TuneCP5_13TeV-pythia8/MINIAODSIM/106X_mc2017_realistic_v9-v1/260000/1D081235-7942-A34B-AC9C-FD7AB2947BF1.root -a '-o ./ -m --year 2017'"
transfer_output_remaps="out_1.root = /ospool/cms-user/yuzhe/NtupleStore/Wcb/2017/MC/WZ_TuneCP5_13TeV-pythia8_RunIISummer20UL17MiniAODv2-106X_mc2017_realistic_v9-v1/out_80.root "
queue
        
arguments="-f /store/mc/RunIISummer20UL17MiniAODv2/WZ_TuneCP5_13TeV-pythia8/MINIAODSIM/106X_mc2017_realistic_v9-v1/40000/7726003B-18FE-9E46-84D8-FFBF51B197A6.root -a '-o ./ -m --year 2017'"
transfer_output_remaps="out_1.root = /ospool/cms-user/yuzhe/NtupleStore/Wcb/2017/MC/WZ_TuneCP5_13TeV-pythia8_RunIISummer20UL17MiniAODv2-106X_mc2017_realistic_v9-v1/out_81.root "
queue
        
arguments="-f /store/mc/RunIISummer20UL17MiniAODv2/WZ_TuneCP5_13TeV-pythia8/MINIAODSIM/106X_mc2017_realistic_v9-v1/40000/3769392E-9FF9-FA45-9499-68FCAB826EFB.root -a '-o ./ -m --year 2017'"
transfer_output_remaps="out_1.root = /ospool/cms-user/yuzhe/NtupleStore/Wcb/2017/MC/WZ_TuneCP5_13TeV-pythia8_RunIISummer20UL17MiniAODv2-106X_mc2017_realistic_v9-v1/out_86.root "
queue
        
arguments="-f /store/mc/RunIISummer20UL17MiniAODv2/WZ_TuneCP5_13TeV-pythia8/MINIAODSIM/106X_mc2017_realistic_v9-v1/40000/5533FB24-5D69-5C46-AB70-7770D9860324.root -a '-o ./ -m --year 2017'"
transfer_output_remaps="out_1.root = /ospool/cms-user/yuzhe/NtupleStore/Wcb/2017/MC/WZ_TuneCP5_13TeV-pythia8_RunIISummer20UL17MiniAODv2-106X_mc2017_realistic_v9-v1/out_87.root "
queue
        
arguments="-f /store/mc/RunIISummer20UL17MiniAODv2/WZ_TuneCP5_13TeV-pythia8/MINIAODSIM/106X_mc2017_realistic_v9-v1/40000/52671DF3-2170-E94F-89E2-E2EF67CA2416.root -a '-o ./ -m --year 2017'"
transfer_output_remaps="out_1.root = /ospool/cms-user/yuzhe/NtupleStore/Wcb/2017/MC/WZ_TuneCP5_13TeV-pythia8_RunIISummer20UL17MiniAODv2-106X_mc2017_realistic_v9-v1/out_84.root "
queue
        
arguments="-f /store/mc/RunIISummer20UL17MiniAODv2/WZ_TuneCP5_13TeV-pythia8/MINIAODSIM/106X_mc2017_realistic_v9-v1/40000/CDEA424E-0821-B145-88EC-EF069128DA9D.root -a '-o ./ -m --year 2017'"
transfer_output_remaps="out_1.root = /ospool/cms-user/yuzhe/NtupleStore/Wcb/2017/MC/WZ_TuneCP5_13TeV-pythia8_RunIISummer20UL17MiniAODv2-106X_mc2017_realistic_v9-v1/out_85.root "
queue
        
arguments="-f /store/mc/RunIISummer20UL17MiniAODv2/WZ_TuneCP5_13TeV-pythia8/MINIAODSIM/106X_mc2017_realistic_v9-v1/260000/55434244-F349-E745-A7C2-0BA4E62D6A8D.root -a '-o ./ -m --year 2017'"
transfer_output_remaps="out_1.root = /ospool/cms-user/yuzhe/NtupleStore/Wcb/2017/MC/WZ_TuneCP5_13TeV-pythia8_RunIISummer20UL17MiniAODv2-106X_mc2017_realistic_v9-v1/out_3.root "
queue
        
arguments="-f /store/mc/RunIISummer20UL17MiniAODv2/WZ_TuneCP5_13TeV-pythia8/MINIAODSIM/106X_mc2017_realistic_v9-v1/50000/7FA7C5F1-4AF6-7F4D-B446-A79CDA1E0C10.root -a '-o ./ -m --year 2017'"
transfer_output_remaps="out_1.root = /ospool/cms-user/yuzhe/NtupleStore/Wcb/2017/MC/WZ_TuneCP5_13TeV-pythia8_RunIISummer20UL17MiniAODv2-106X_mc2017_realistic_v9-v1/out_7.root "
queue
        
arguments="-f /store/mc/RunIISummer20UL17MiniAODv2/WZ_TuneCP5_13TeV-pythia8/MINIAODSIM/106X_mc2017_realistic_v9-v1/50000/9358CB0D-7F85-294B-8E49-BA248061F523.root -a '-o ./ -m --year 2017'"
transfer_output_remaps="out_1.root = /ospool/cms-user/yuzhe/NtupleStore/Wcb/2017/MC/WZ_TuneCP5_13TeV-pythia8_RunIISummer20UL17MiniAODv2-106X_mc2017_realistic_v9-v1/out_102.root "
queue
        
arguments="-f /store/mc/RunIISummer20UL17MiniAODv2/WZ_TuneCP5_13TeV-pythia8/MINIAODSIM/106X_mc2017_realistic_v9-v1/50000/6B83B648-A171-F74F-AADC-E4FAB6E2AEDD.root -a '-o ./ -m --year 2017'"
transfer_output_remaps="out_1.root = /ospool/cms-user/yuzhe/NtupleStore/Wcb/2017/MC/WZ_TuneCP5_13TeV-pythia8_RunIISummer20UL17MiniAODv2-106X_mc2017_realistic_v9-v1/out_103.root "
queue
        
arguments="-f /store/mc/RunIISummer20UL17MiniAODv2/WZ_TuneCP5_13TeV-pythia8/MINIAODSIM/106X_mc2017_realistic_v9-v1/50000/E6115AC3-3315-2C4D-B625-C420658BD4DD.root -a '-o ./ -m --year 2017'"
transfer_output_remaps="out_1.root = /ospool/cms-user/yuzhe/NtupleStore/Wcb/2017/MC/WZ_TuneCP5_13TeV-pythia8_RunIISummer20UL17MiniAODv2-106X_mc2017_realistic_v9-v1/out_100.root "
queue
        
arguments="-f /store/mc/RunIISummer20UL17MiniAODv2/WZ_TuneCP5_13TeV-pythia8/MINIAODSIM/106X_mc2017_realistic_v9-v1/50000/AE06B1EA-7DFB-8942-87B5-20009D0001FC.root -a '-o ./ -m --year 2017'"
transfer_output_remaps="out_1.root = /ospool/cms-user/yuzhe/NtupleStore/Wcb/2017/MC/WZ_TuneCP5_13TeV-pythia8_RunIISummer20UL17MiniAODv2-106X_mc2017_realistic_v9-v1/out_101.root "
queue
        
arguments="-f /store/mc/RunIISummer20UL17MiniAODv2/WZ_TuneCP5_13TeV-pythia8/MINIAODSIM/106X_mc2017_realistic_v9-v1/50000/8BCF1C37-EC5A-CB4B-851A-62EA5DEC423F.root -a '-o ./ -m --year 2017'"
transfer_output_remaps="out_1.root = /ospool/cms-user/yuzhe/NtupleStore/Wcb/2017/MC/WZ_TuneCP5_13TeV-pythia8_RunIISummer20UL17MiniAODv2-106X_mc2017_realistic_v9-v1/out_104.root "
queue
        
arguments="-f /store/mc/RunIISummer20UL17MiniAODv2/WZ_TuneCP5_13TeV-pythia8/MINIAODSIM/106X_mc2017_realistic_v9-v1/50000/91FD0510-EC9F-0C4C-8927-2E610855F892.root -a '-o ./ -m --year 2017'"
transfer_output_remaps="out_1.root = /ospool/cms-user/yuzhe/NtupleStore/Wcb/2017/MC/WZ_TuneCP5_13TeV-pythia8_RunIISummer20UL17MiniAODv2-106X_mc2017_realistic_v9-v1/out_105.root "
queue
        
arguments="-f /store/mc/RunIISummer20UL17MiniAODv2/WZ_TuneCP5_13TeV-pythia8/MINIAODSIM/106X_mc2017_realistic_v9-v1/280000/5873F760-843C-EC48-965C-6C03719E406C.root -a '-o ./ -m --year 2017'"
transfer_output_remaps="out_1.root = /ospool/cms-user/yuzhe/NtupleStore/Wcb/2017/MC/WZ_TuneCP5_13TeV-pythia8_RunIISummer20UL17MiniAODv2-106X_mc2017_realistic_v9-v1/out_39.root "
queue
        
arguments="-f /store/mc/RunIISummer20UL17MiniAODv2/WZ_TuneCP5_13TeV-pythia8/MINIAODSIM/106X_mc2017_realistic_v9-v1/280000/7648CB7E-2479-0C46-AA70-C9AD5B681095.root -a '-o ./ -m --year 2017'"
transfer_output_remaps="out_1.root = /ospool/cms-user/yuzhe/NtupleStore/Wcb/2017/MC/WZ_TuneCP5_13TeV-pythia8_RunIISummer20UL17MiniAODv2-106X_mc2017_realistic_v9-v1/out_38.root "
queue
        
arguments="-f /store/mc/RunIISummer20UL17MiniAODv2/WZ_TuneCP5_13TeV-pythia8/MINIAODSIM/106X_mc2017_realistic_v9-v1/280000/76ED3EAD-C4FE-E74D-801B-4B44DF393C6C.root -a '-o ./ -m --year 2017'"
transfer_output_remaps="out_1.root = /ospool/cms-user/yuzhe/NtupleStore/Wcb/2017/MC/WZ_TuneCP5_13TeV-pythia8_RunIISummer20UL17MiniAODv2-106X_mc2017_realistic_v9-v1/out_33.root "
queue
        
arguments="-f /store/mc/RunIISummer20UL17MiniAODv2/WZ_TuneCP5_13TeV-pythia8/MINIAODSIM/106X_mc2017_realistic_v9-v1/280000/94F57220-EE55-BF4B-A486-B46077BE5B56.root -a '-o ./ -m --year 2017'"
transfer_output_remaps="out_1.root = /ospool/cms-user/yuzhe/NtupleStore/Wcb/2017/MC/WZ_TuneCP5_13TeV-pythia8_RunIISummer20UL17MiniAODv2-106X_mc2017_realistic_v9-v1/out_32.root "
queue
        
arguments="-f /store/mc/RunIISummer20UL17MiniAODv2/WZ_TuneCP5_13TeV-pythia8/MINIAODSIM/106X_mc2017_realistic_v9-v1/280000/7B0ADAD3-FFBF-324E-A287-66907D929CAE.root -a '-o ./ -m --year 2017'"
transfer_output_remaps="out_1.root = /ospool/cms-user/yuzhe/NtupleStore/Wcb/2017/MC/WZ_TuneCP5_13TeV-pythia8_RunIISummer20UL17MiniAODv2-106X_mc2017_realistic_v9-v1/out_31.root "
queue
        
arguments="-f /store/mc/RunIISummer20UL17MiniAODv2/WZ_TuneCP5_13TeV-pythia8/MINIAODSIM/106X_mc2017_realistic_v9-v1/280000/9921CF0D-3DF6-C544-96A1-709FDC96B889.root -a '-o ./ -m --year 2017'"
transfer_output_remaps="out_1.root = /ospool/cms-user/yuzhe/NtupleStore/Wcb/2017/MC/WZ_TuneCP5_13TeV-pythia8_RunIISummer20UL17MiniAODv2-106X_mc2017_realistic_v9-v1/out_30.root "
queue
        
arguments="-f /store/mc/RunIISummer20UL17MiniAODv2/WZ_TuneCP5_13TeV-pythia8/MINIAODSIM/106X_mc2017_realistic_v9-v1/280000/1DF7F357-6CF5-8B42-B9A4-89E314526559.root -a '-o ./ -m --year 2017'"
transfer_output_remaps="out_1.root = /ospool/cms-user/yuzhe/NtupleStore/Wcb/2017/MC/WZ_TuneCP5_13TeV-pythia8_RunIISummer20UL17MiniAODv2-106X_mc2017_realistic_v9-v1/out_37.root "
queue
        
arguments="-f /store/mc/RunIISummer20UL17MiniAODv2/WZ_TuneCP5_13TeV-pythia8/MINIAODSIM/106X_mc2017_realistic_v9-v1/280000/742C512B-BEE8-B743-BC53-30602B6BDD21.root -a '-o ./ -m --year 2017'"
transfer_output_remaps="out_1.root = /ospool/cms-user/yuzhe/NtupleStore/Wcb/2017/MC/WZ_TuneCP5_13TeV-pythia8_RunIISummer20UL17MiniAODv2-106X_mc2017_realistic_v9-v1/out_36.root "
queue
        
arguments="-f /store/mc/RunIISummer20UL17MiniAODv2/WZ_TuneCP5_13TeV-pythia8/MINIAODSIM/106X_mc2017_realistic_v9-v1/280000/53119C1D-424E-E14C-95DB-82A95CBD4AA8.root -a '-o ./ -m --year 2017'"
transfer_output_remaps="out_1.root = /ospool/cms-user/yuzhe/NtupleStore/Wcb/2017/MC/WZ_TuneCP5_13TeV-pythia8_RunIISummer20UL17MiniAODv2-106X_mc2017_realistic_v9-v1/out_35.root "
queue
        
arguments="-f /store/mc/RunIISummer20UL17MiniAODv2/WZ_TuneCP5_13TeV-pythia8/MINIAODSIM/106X_mc2017_realistic_v9-v1/280000/D6BB0F86-A8FB-DF43-B5D1-937AA375E46A.root -a '-o ./ -m --year 2017'"
transfer_output_remaps="out_1.root = /ospool/cms-user/yuzhe/NtupleStore/Wcb/2017/MC/WZ_TuneCP5_13TeV-pythia8_RunIISummer20UL17MiniAODv2-106X_mc2017_realistic_v9-v1/out_34.root "
queue
        
arguments="-f /store/mc/RunIISummer20UL17MiniAODv2/WZ_TuneCP5_13TeV-pythia8/MINIAODSIM/106X_mc2017_realistic_v9-v1/280000/C296A9F0-3035-5342-9069-BD095538070B.root -a '-o ./ -m --year 2017'"
transfer_output_remaps="out_1.root = /ospool/cms-user/yuzhe/NtupleStore/Wcb/2017/MC/WZ_TuneCP5_13TeV-pythia8_RunIISummer20UL17MiniAODv2-106X_mc2017_realistic_v9-v1/out_60.root "
queue
        
arguments="-f /store/mc/RunIISummer20UL17MiniAODv2/WZ_TuneCP5_13TeV-pythia8/MINIAODSIM/106X_mc2017_realistic_v9-v1/280000/F43BBDFB-2F6F-224E-8B5C-B349A9A90170.root -a '-o ./ -m --year 2017'"
transfer_output_remaps="out_1.root = /ospool/cms-user/yuzhe/NtupleStore/Wcb/2017/MC/WZ_TuneCP5_13TeV-pythia8_RunIISummer20UL17MiniAODv2-106X_mc2017_realistic_v9-v1/out_61.root "
queue
        
arguments="-f /store/mc/RunIISummer20UL17MiniAODv2/WZ_TuneCP5_13TeV-pythia8/MINIAODSIM/106X_mc2017_realistic_v9-v1/280000/44A1BB8D-0729-894D-B123-F22908FD3F68.root -a '-o ./ -m --year 2017'"
transfer_output_remaps="out_1.root = /ospool/cms-user/yuzhe/NtupleStore/Wcb/2017/MC/WZ_TuneCP5_13TeV-pythia8_RunIISummer20UL17MiniAODv2-106X_mc2017_realistic_v9-v1/out_62.root "
queue
        
arguments="-f /store/mc/RunIISummer20UL17MiniAODv2/WZ_TuneCP5_13TeV-pythia8/MINIAODSIM/106X_mc2017_realistic_v9-v1/280000/90BA6C6F-99CB-694B-971F-16C848978B80.root -a '-o ./ -m --year 2017'"
transfer_output_remaps="out_1.root = /ospool/cms-user/yuzhe/NtupleStore/Wcb/2017/MC/WZ_TuneCP5_13TeV-pythia8_RunIISummer20UL17MiniAODv2-106X_mc2017_realistic_v9-v1/out_63.root "
queue
        
arguments="-f /store/mc/RunIISummer20UL17MiniAODv2/WZ_TuneCP5_13TeV-pythia8/MINIAODSIM/106X_mc2017_realistic_v9-v1/280000/3110B27C-77FE-1345-86E1-8C1DD35A81FE.root -a '-o ./ -m --year 2017'"
transfer_output_remaps="out_1.root = /ospool/cms-user/yuzhe/NtupleStore/Wcb/2017/MC/WZ_TuneCP5_13TeV-pythia8_RunIISummer20UL17MiniAODv2-106X_mc2017_realistic_v9-v1/out_64.root "
queue
        
arguments="-f /store/mc/RunIISummer20UL17MiniAODv2/WZ_TuneCP5_13TeV-pythia8/MINIAODSIM/106X_mc2017_realistic_v9-v1/280000/21A38C8E-1801-6240-BBBE-95850765F9A7.root -a '-o ./ -m --year 2017'"
transfer_output_remaps="out_1.root = /ospool/cms-user/yuzhe/NtupleStore/Wcb/2017/MC/WZ_TuneCP5_13TeV-pythia8_RunIISummer20UL17MiniAODv2-106X_mc2017_realistic_v9-v1/out_65.root "
queue
        
arguments="-f /store/mc/RunIISummer20UL17MiniAODv2/WZ_TuneCP5_13TeV-pythia8/MINIAODSIM/106X_mc2017_realistic_v9-v1/280000/15C07164-E80E-1644-B1E2-0A6520B2A9FB.root -a '-o ./ -m --year 2017'"
transfer_output_remaps="out_1.root = /ospool/cms-user/yuzhe/NtupleStore/Wcb/2017/MC/WZ_TuneCP5_13TeV-pythia8_RunIISummer20UL17MiniAODv2-106X_mc2017_realistic_v9-v1/out_66.root "
queue
        
arguments="-f /store/mc/RunIISummer20UL17MiniAODv2/WZ_TuneCP5_13TeV-pythia8/MINIAODSIM/106X_mc2017_realistic_v9-v1/280000/166F5FBD-695F-264F-83C1-790C79B8F5DC.root -a '-o ./ -m --year 2017'"
transfer_output_remaps="out_1.root = /ospool/cms-user/yuzhe/NtupleStore/Wcb/2017/MC/WZ_TuneCP5_13TeV-pythia8_RunIISummer20UL17MiniAODv2-106X_mc2017_realistic_v9-v1/out_67.root "
queue
        
arguments="-f /store/mc/RunIISummer20UL17MiniAODv2/WZ_TuneCP5_13TeV-pythia8/MINIAODSIM/106X_mc2017_realistic_v9-v1/280000/D17440EA-B6BF-2943-9820-56C847D3BFDC.root -a '-o ./ -m --year 2017'"
transfer_output_remaps="out_1.root = /ospool/cms-user/yuzhe/NtupleStore/Wcb/2017/MC/WZ_TuneCP5_13TeV-pythia8_RunIISummer20UL17MiniAODv2-106X_mc2017_realistic_v9-v1/out_68.root "
queue
        
arguments="-f /store/mc/RunIISummer20UL17MiniAODv2/WZ_TuneCP5_13TeV-pythia8/MINIAODSIM/106X_mc2017_realistic_v9-v1/280000/209EDF4E-DA28-E041-8232-C73BFBEDBE72.root -a '-o ./ -m --year 2017'"
transfer_output_remaps="out_1.root = /ospool/cms-user/yuzhe/NtupleStore/Wcb/2017/MC/WZ_TuneCP5_13TeV-pythia8_RunIISummer20UL17MiniAODv2-106X_mc2017_realistic_v9-v1/out_69.root "
queue
        
arguments="-f /store/mc/RunIISummer20UL17MiniAODv2/WZ_TuneCP5_13TeV-pythia8/MINIAODSIM/106X_mc2017_realistic_v9-v1/50000/7E0B8771-8842-8743-B53F-63B82DE64747.root -a '-o ./ -m --year 2017'"
transfer_output_remaps="out_1.root = /ospool/cms-user/yuzhe/NtupleStore/Wcb/2017/MC/WZ_TuneCP5_13TeV-pythia8_RunIISummer20UL17MiniAODv2-106X_mc2017_realistic_v9-v1/out_2.root "
queue
        
arguments="-f /store/mc/RunIISummer20UL17MiniAODv2/WZ_TuneCP5_13TeV-pythia8/MINIAODSIM/106X_mc2017_realistic_v9-v1/50000/A3BF756D-64F6-F046-91E4-90E76B51F78A.root -a '-o ./ -m --year 2017'"
transfer_output_remaps="out_1.root = /ospool/cms-user/yuzhe/NtupleStore/Wcb/2017/MC/WZ_TuneCP5_13TeV-pythia8_RunIISummer20UL17MiniAODv2-106X_mc2017_realistic_v9-v1/out_6.root "
queue
        
arguments="-f /store/mc/RunIISummer20UL17MiniAODv2/WZ_TuneCP5_13TeV-pythia8/MINIAODSIM/106X_mc2017_realistic_v9-v1/50000/24EDC050-8B88-504B-A861-796D3A6F4DA1.root -a '-o ./ -m --year 2017'"
transfer_output_remaps="out_1.root = /ospool/cms-user/yuzhe/NtupleStore/Wcb/2017/MC/WZ_TuneCP5_13TeV-pythia8_RunIISummer20UL17MiniAODv2-106X_mc2017_realistic_v9-v1/out_99.root "
queue
        
arguments="-f /store/mc/RunIISummer20UL17MiniAODv2/WZ_TuneCP5_13TeV-pythia8/MINIAODSIM/106X_mc2017_realistic_v9-v1/50000/87CB24C3-20B5-5844-BBF5-2130A91E5F3A.root -a '-o ./ -m --year 2017'"
transfer_output_remaps="out_1.root = /ospool/cms-user/yuzhe/NtupleStore/Wcb/2017/MC/WZ_TuneCP5_13TeV-pythia8_RunIISummer20UL17MiniAODv2-106X_mc2017_realistic_v9-v1/out_98.root "
queue
        
arguments="-f /store/mc/RunIISummer20UL17MiniAODv2/WZ_TuneCP5_13TeV-pythia8/MINIAODSIM/106X_mc2017_realistic_v9-v1/40000/13731FA6-D7E9-2F44-B617-A6B6903DF98D.root -a '-o ./ -m --year 2017'"
transfer_output_remaps="out_1.root = /ospool/cms-user/yuzhe/NtupleStore/Wcb/2017/MC/WZ_TuneCP5_13TeV-pythia8_RunIISummer20UL17MiniAODv2-106X_mc2017_realistic_v9-v1/out_91.root "
queue
        
arguments="-f /store/mc/RunIISummer20UL17MiniAODv2/WZ_TuneCP5_13TeV-pythia8/MINIAODSIM/106X_mc2017_realistic_v9-v1/40000/693EC9E1-52BF-924E-89F8-3479C5827D68.root -a '-o ./ -m --year 2017'"
transfer_output_remaps="out_1.root = /ospool/cms-user/yuzhe/NtupleStore/Wcb/2017/MC/WZ_TuneCP5_13TeV-pythia8_RunIISummer20UL17MiniAODv2-106X_mc2017_realistic_v9-v1/out_90.root "
queue
        
arguments="-f /store/mc/RunIISummer20UL17MiniAODv2/WZ_TuneCP5_13TeV-pythia8/MINIAODSIM/106X_mc2017_realistic_v9-v1/40000/6C6DCA73-1BFF-2049-B106-37F18D0B40A9.root -a '-o ./ -m --year 2017'"
transfer_output_remaps="out_1.root = /ospool/cms-user/yuzhe/NtupleStore/Wcb/2017/MC/WZ_TuneCP5_13TeV-pythia8_RunIISummer20UL17MiniAODv2-106X_mc2017_realistic_v9-v1/out_93.root "
queue
        
arguments="-f /store/mc/RunIISummer20UL17MiniAODv2/WZ_TuneCP5_13TeV-pythia8/MINIAODSIM/106X_mc2017_realistic_v9-v1/40000/1B1B7F94-C387-E041-941E-63787F754183.root -a '-o ./ -m --year 2017'"
transfer_output_remaps="out_1.root = /ospool/cms-user/yuzhe/NtupleStore/Wcb/2017/MC/WZ_TuneCP5_13TeV-pythia8_RunIISummer20UL17MiniAODv2-106X_mc2017_realistic_v9-v1/out_92.root "
queue
        
arguments="-f /store/mc/RunIISummer20UL17MiniAODv2/WZ_TuneCP5_13TeV-pythia8/MINIAODSIM/106X_mc2017_realistic_v9-v1/50000/3C3A8EC2-9AB0-D048-BCF4-352D94EBCFA1.root -a '-o ./ -m --year 2017'"
transfer_output_remaps="out_1.root = /ospool/cms-user/yuzhe/NtupleStore/Wcb/2017/MC/WZ_TuneCP5_13TeV-pythia8_RunIISummer20UL17MiniAODv2-106X_mc2017_realistic_v9-v1/out_95.root "
queue
        
arguments="-f /store/mc/RunIISummer20UL17MiniAODv2/WZ_TuneCP5_13TeV-pythia8/MINIAODSIM/106X_mc2017_realistic_v9-v1/40000/424EEE4B-6202-084D-AC94-BFC97E86980E.root -a '-o ./ -m --year 2017'"
transfer_output_remaps="out_1.root = /ospool/cms-user/yuzhe/NtupleStore/Wcb/2017/MC/WZ_TuneCP5_13TeV-pythia8_RunIISummer20UL17MiniAODv2-106X_mc2017_realistic_v9-v1/out_94.root "
queue
        
arguments="-f /store/mc/RunIISummer20UL17MiniAODv2/WZ_TuneCP5_13TeV-pythia8/MINIAODSIM/106X_mc2017_realistic_v9-v1/50000/1E987689-A879-6645-9978-92B525E63A4A.root -a '-o ./ -m --year 2017'"
transfer_output_remaps="out_1.root = /ospool/cms-user/yuzhe/NtupleStore/Wcb/2017/MC/WZ_TuneCP5_13TeV-pythia8_RunIISummer20UL17MiniAODv2-106X_mc2017_realistic_v9-v1/out_97.root "
queue
        
arguments="-f /store/mc/RunIISummer20UL17MiniAODv2/WZ_TuneCP5_13TeV-pythia8/MINIAODSIM/106X_mc2017_realistic_v9-v1/50000/46ED68D8-1EAB-4E46-8A98-985774E9FCEC.root -a '-o ./ -m --year 2017'"
transfer_output_remaps="out_1.root = /ospool/cms-user/yuzhe/NtupleStore/Wcb/2017/MC/WZ_TuneCP5_13TeV-pythia8_RunIISummer20UL17MiniAODv2-106X_mc2017_realistic_v9-v1/out_96.root "
queue
        
arguments="-f /store/mc/RunIISummer20UL17MiniAODv2/WZ_TuneCP5_13TeV-pythia8/MINIAODSIM/106X_mc2017_realistic_v9-v1/270000/85448812-27E8-C941-9A68-BD01331F3656.root -a '-o ./ -m --year 2017'"
transfer_output_remaps="out_1.root = /ospool/cms-user/yuzhe/NtupleStore/Wcb/2017/MC/WZ_TuneCP5_13TeV-pythia8_RunIISummer20UL17MiniAODv2-106X_mc2017_realistic_v9-v1/out_11.root "
queue
        
arguments="-f /store/mc/RunIISummer20UL17MiniAODv2/WZ_TuneCP5_13TeV-pythia8/MINIAODSIM/106X_mc2017_realistic_v9-v1/50000/09E77982-2387-7C4E-887C-C6BDFE19DAEE.root -a '-o ./ -m --year 2017'"
transfer_output_remaps="out_1.root = /ospool/cms-user/yuzhe/NtupleStore/Wcb/2017/MC/WZ_TuneCP5_13TeV-pythia8_RunIISummer20UL17MiniAODv2-106X_mc2017_realistic_v9-v1/out_10.root "
queue
        
arguments="-f /store/mc/RunIISummer20UL17MiniAODv2/WZ_TuneCP5_13TeV-pythia8/MINIAODSIM/106X_mc2017_realistic_v9-v1/270000/5D62A56A-9CB2-6341-89A6-6676ABF77CD1.root -a '-o ./ -m --year 2017'"
transfer_output_remaps="out_1.root = /ospool/cms-user/yuzhe/NtupleStore/Wcb/2017/MC/WZ_TuneCP5_13TeV-pythia8_RunIISummer20UL17MiniAODv2-106X_mc2017_realistic_v9-v1/out_13.root "
queue
        
arguments="-f /store/mc/RunIISummer20UL17MiniAODv2/WZ_TuneCP5_13TeV-pythia8/MINIAODSIM/106X_mc2017_realistic_v9-v1/270000/EEA2499E-6631-4040-9D03-43F0AC813AAB.root -a '-o ./ -m --year 2017'"
transfer_output_remaps="out_1.root = /ospool/cms-user/yuzhe/NtupleStore/Wcb/2017/MC/WZ_TuneCP5_13TeV-pythia8_RunIISummer20UL17MiniAODv2-106X_mc2017_realistic_v9-v1/out_12.root "
queue
        
arguments="-f /store/mc/RunIISummer20UL17MiniAODv2/WZ_TuneCP5_13TeV-pythia8/MINIAODSIM/106X_mc2017_realistic_v9-v1/270000/78E0750D-89EE-774F-93E0-848625FF5CB1.root -a '-o ./ -m --year 2017'"
transfer_output_remaps="out_1.root = /ospool/cms-user/yuzhe/NtupleStore/Wcb/2017/MC/WZ_TuneCP5_13TeV-pythia8_RunIISummer20UL17MiniAODv2-106X_mc2017_realistic_v9-v1/out_15.root "
queue
        
arguments="-f /store/mc/RunIISummer20UL17MiniAODv2/WZ_TuneCP5_13TeV-pythia8/MINIAODSIM/106X_mc2017_realistic_v9-v1/270000/6E3F9691-647A-674D-A0E0-6BC91E0F66E2.root -a '-o ./ -m --year 2017'"
transfer_output_remaps="out_1.root = /ospool/cms-user/yuzhe/NtupleStore/Wcb/2017/MC/WZ_TuneCP5_13TeV-pythia8_RunIISummer20UL17MiniAODv2-106X_mc2017_realistic_v9-v1/out_14.root "
queue
        
arguments="-f /store/mc/RunIISummer20UL17MiniAODv2/WZ_TuneCP5_13TeV-pythia8/MINIAODSIM/106X_mc2017_realistic_v9-v1/280000/492B3692-6F70-A44F-A5D4-587170890705.root -a '-o ./ -m --year 2017'"
transfer_output_remaps="out_1.root = /ospool/cms-user/yuzhe/NtupleStore/Wcb/2017/MC/WZ_TuneCP5_13TeV-pythia8_RunIISummer20UL17MiniAODv2-106X_mc2017_realistic_v9-v1/out_17.root "
queue
        
arguments="-f /store/mc/RunIISummer20UL17MiniAODv2/WZ_TuneCP5_13TeV-pythia8/MINIAODSIM/106X_mc2017_realistic_v9-v1/50000/591EC5BF-6AE4-DD44-8744-5496B2CD8DAB.root -a '-o ./ -m --year 2017'"
transfer_output_remaps="out_1.root = /ospool/cms-user/yuzhe/NtupleStore/Wcb/2017/MC/WZ_TuneCP5_13TeV-pythia8_RunIISummer20UL17MiniAODv2-106X_mc2017_realistic_v9-v1/out_16.root "
queue
        
arguments="-f /store/mc/RunIISummer20UL17MiniAODv2/WZ_TuneCP5_13TeV-pythia8/MINIAODSIM/106X_mc2017_realistic_v9-v1/280000/B5D713B4-1E20-8041-94A0-3A0CFB31A924.root -a '-o ./ -m --year 2017'"
transfer_output_remaps="out_1.root = /ospool/cms-user/yuzhe/NtupleStore/Wcb/2017/MC/WZ_TuneCP5_13TeV-pythia8_RunIISummer20UL17MiniAODv2-106X_mc2017_realistic_v9-v1/out_19.root "
queue
        
arguments="-f /store/mc/RunIISummer20UL17MiniAODv2/WZ_TuneCP5_13TeV-pythia8/MINIAODSIM/106X_mc2017_realistic_v9-v1/280000/EB963E68-600B-784B-ACFF-30F584941294.root -a '-o ./ -m --year 2017'"
transfer_output_remaps="out_1.root = /ospool/cms-user/yuzhe/NtupleStore/Wcb/2017/MC/WZ_TuneCP5_13TeV-pythia8_RunIISummer20UL17MiniAODv2-106X_mc2017_realistic_v9-v1/out_18.root "
queue
        
arguments="-f /store/mc/RunIISummer20UL17MiniAODv2/WZ_TuneCP5_13TeV-pythia8/MINIAODSIM/106X_mc2017_realistic_v9-v1/280000/72F742D0-BA19-EE40-81A2-AABE9E3A6EB0.root -a '-o ./ -m --year 2017'"
transfer_output_remaps="out_1.root = /ospool/cms-user/yuzhe/NtupleStore/Wcb/2017/MC/WZ_TuneCP5_13TeV-pythia8_RunIISummer20UL17MiniAODv2-106X_mc2017_realistic_v9-v1/out_48.root "
queue
        
arguments="-f /store/mc/RunIISummer20UL17MiniAODv2/WZ_TuneCP5_13TeV-pythia8/MINIAODSIM/106X_mc2017_realistic_v9-v1/280000/B248A54B-843A-B443-BAAE-00436A6E1ABD.root -a '-o ./ -m --year 2017'"
transfer_output_remaps="out_1.root = /ospool/cms-user/yuzhe/NtupleStore/Wcb/2017/MC/WZ_TuneCP5_13TeV-pythia8_RunIISummer20UL17MiniAODv2-106X_mc2017_realistic_v9-v1/out_49.root "
queue
        
arguments="-f /store/mc/RunIISummer20UL17MiniAODv2/WZ_TuneCP5_13TeV-pythia8/MINIAODSIM/106X_mc2017_realistic_v9-v1/280000/D2DF1BCA-C147-3E4B-9EE5-1CF70C920166.root -a '-o ./ -m --year 2017'"
transfer_output_remaps="out_1.root = /ospool/cms-user/yuzhe/NtupleStore/Wcb/2017/MC/WZ_TuneCP5_13TeV-pythia8_RunIISummer20UL17MiniAODv2-106X_mc2017_realistic_v9-v1/out_46.root "
queue
        
arguments="-f /store/mc/RunIISummer20UL17MiniAODv2/WZ_TuneCP5_13TeV-pythia8/MINIAODSIM/106X_mc2017_realistic_v9-v1/280000/6D683827-189B-4E49-9751-FE6F2EBBD829.root -a '-o ./ -m --year 2017'"
transfer_output_remaps="out_1.root = /ospool/cms-user/yuzhe/NtupleStore/Wcb/2017/MC/WZ_TuneCP5_13TeV-pythia8_RunIISummer20UL17MiniAODv2-106X_mc2017_realistic_v9-v1/out_47.root "
queue
        
arguments="-f /store/mc/RunIISummer20UL17MiniAODv2/WZ_TuneCP5_13TeV-pythia8/MINIAODSIM/106X_mc2017_realistic_v9-v1/280000/11351566-6712-0B4D-B469-2CCBA2FBFB44.root -a '-o ./ -m --year 2017'"
transfer_output_remaps="out_1.root = /ospool/cms-user/yuzhe/NtupleStore/Wcb/2017/MC/WZ_TuneCP5_13TeV-pythia8_RunIISummer20UL17MiniAODv2-106X_mc2017_realistic_v9-v1/out_44.root "
queue
        
arguments="-f /store/mc/RunIISummer20UL17MiniAODv2/WZ_TuneCP5_13TeV-pythia8/MINIAODSIM/106X_mc2017_realistic_v9-v1/280000/6DB8B057-5DDC-6847-80D6-ACBE43B0D0F3.root -a '-o ./ -m --year 2017'"
transfer_output_remaps="out_1.root = /ospool/cms-user/yuzhe/NtupleStore/Wcb/2017/MC/WZ_TuneCP5_13TeV-pythia8_RunIISummer20UL17MiniAODv2-106X_mc2017_realistic_v9-v1/out_45.root "
queue
        
arguments="-f /store/mc/RunIISummer20UL17MiniAODv2/WZ_TuneCP5_13TeV-pythia8/MINIAODSIM/106X_mc2017_realistic_v9-v1/280000/F802A8FF-9C9B-8047-B3C5-1400B3100AB7.root -a '-o ./ -m --year 2017'"
transfer_output_remaps="out_1.root = /ospool/cms-user/yuzhe/NtupleStore/Wcb/2017/MC/WZ_TuneCP5_13TeV-pythia8_RunIISummer20UL17MiniAODv2-106X_mc2017_realistic_v9-v1/out_42.root "
queue
        
arguments="-f /store/mc/RunIISummer20UL17MiniAODv2/WZ_TuneCP5_13TeV-pythia8/MINIAODSIM/106X_mc2017_realistic_v9-v1/280000/053F11D5-3B68-EA48-9F60-3FDC3E6A1F6E.root -a '-o ./ -m --year 2017'"
transfer_output_remaps="out_1.root = /ospool/cms-user/yuzhe/NtupleStore/Wcb/2017/MC/WZ_TuneCP5_13TeV-pythia8_RunIISummer20UL17MiniAODv2-106X_mc2017_realistic_v9-v1/out_43.root "
queue
        
arguments="-f /store/mc/RunIISummer20UL17MiniAODv2/WZ_TuneCP5_13TeV-pythia8/MINIAODSIM/106X_mc2017_realistic_v9-v1/280000/2FC5F009-329F-B94A-9843-49FCFDF74A51.root -a '-o ./ -m --year 2017'"
transfer_output_remaps="out_1.root = /ospool/cms-user/yuzhe/NtupleStore/Wcb/2017/MC/WZ_TuneCP5_13TeV-pythia8_RunIISummer20UL17MiniAODv2-106X_mc2017_realistic_v9-v1/out_40.root "
queue
        
arguments="-f /store/mc/RunIISummer20UL17MiniAODv2/WZ_TuneCP5_13TeV-pythia8/MINIAODSIM/106X_mc2017_realistic_v9-v1/280000/AB967CC5-7EB5-FF4B-A9B8-1E94D76743CF.root -a '-o ./ -m --year 2017'"
transfer_output_remaps="out_1.root = /ospool/cms-user/yuzhe/NtupleStore/Wcb/2017/MC/WZ_TuneCP5_13TeV-pythia8_RunIISummer20UL17MiniAODv2-106X_mc2017_realistic_v9-v1/out_41.root "
queue
        
arguments="-f /store/mc/RunIISummer20UL17MiniAODv2/WZ_TuneCP5_13TeV-pythia8/MINIAODSIM/106X_mc2017_realistic_v9-v1/50000/716D2B20-D52B-4143-BB0E-77CD652CC60A.root -a '-o ./ -m --year 2017'"
transfer_output_remaps="out_1.root = /ospool/cms-user/yuzhe/NtupleStore/Wcb/2017/MC/WZ_TuneCP5_13TeV-pythia8_RunIISummer20UL17MiniAODv2-106X_mc2017_realistic_v9-v1/out_1.root "
queue
        
arguments="-f /store/mc/RunIISummer20UL17MiniAODv2/WZ_TuneCP5_13TeV-pythia8/MINIAODSIM/106X_mc2017_realistic_v9-v1/50000/089A5A79-C914-9443-AE77-BF3E8D6D1711.root -a '-o ./ -m --year 2017'"
transfer_output_remaps="out_1.root = /ospool/cms-user/yuzhe/NtupleStore/Wcb/2017/MC/WZ_TuneCP5_13TeV-pythia8_RunIISummer20UL17MiniAODv2-106X_mc2017_realistic_v9-v1/out_5.root "
queue
        
arguments="-f /store/mc/RunIISummer20UL17MiniAODv2/WZ_TuneCP5_13TeV-pythia8/MINIAODSIM/106X_mc2017_realistic_v9-v1/280000/7C4CC084-3613-5245-9DBB-E9D449937B8E.root -a '-o ./ -m --year 2017'"
transfer_output_remaps="out_1.root = /ospool/cms-user/yuzhe/NtupleStore/Wcb/2017/MC/WZ_TuneCP5_13TeV-pythia8_RunIISummer20UL17MiniAODv2-106X_mc2017_realistic_v9-v1/out_9.root "
queue
        
arguments="-f /store/mc/RunIISummer20UL17MiniAODv2/WZ_TuneCP5_13TeV-pythia8/MINIAODSIM/106X_mc2017_realistic_v9-v1/280000/EBC5B3E2-9486-A44C-952E-B5A9EF88158A.root -a '-o ./ -m --year 2017'"
transfer_output_remaps="out_1.root = /ospool/cms-user/yuzhe/NtupleStore/Wcb/2017/MC/WZ_TuneCP5_13TeV-pythia8_RunIISummer20UL17MiniAODv2-106X_mc2017_realistic_v9-v1/out_77.root "
queue
        
arguments="-f /store/mc/RunIISummer20UL17MiniAODv2/WZ_TuneCP5_13TeV-pythia8/MINIAODSIM/106X_mc2017_realistic_v9-v1/280000/F5DF28C0-2DEC-6B43-A274-442D988D0AB8.root -a '-o ./ -m --year 2017'"
transfer_output_remaps="out_1.root = /ospool/cms-user/yuzhe/NtupleStore/Wcb/2017/MC/WZ_TuneCP5_13TeV-pythia8_RunIISummer20UL17MiniAODv2-106X_mc2017_realistic_v9-v1/out_76.root "
queue
        
arguments="-f /store/mc/RunIISummer20UL17MiniAODv2/WZ_TuneCP5_13TeV-pythia8/MINIAODSIM/106X_mc2017_realistic_v9-v1/280000/74210236-B1C2-014E-8DE0-12E3475B9163.root -a '-o ./ -m --year 2017'"
transfer_output_remaps="out_1.root = /ospool/cms-user/yuzhe/NtupleStore/Wcb/2017/MC/WZ_TuneCP5_13TeV-pythia8_RunIISummer20UL17MiniAODv2-106X_mc2017_realistic_v9-v1/out_75.root "
queue
        
arguments="-f /store/mc/RunIISummer20UL17MiniAODv2/WZ_TuneCP5_13TeV-pythia8/MINIAODSIM/106X_mc2017_realistic_v9-v1/280000/044C310A-7B64-854E-87DC-236717F1D278.root -a '-o ./ -m --year 2017'"
transfer_output_remaps="out_1.root = /ospool/cms-user/yuzhe/NtupleStore/Wcb/2017/MC/WZ_TuneCP5_13TeV-pythia8_RunIISummer20UL17MiniAODv2-106X_mc2017_realistic_v9-v1/out_74.root "
queue
        
arguments="-f /store/mc/RunIISummer20UL17MiniAODv2/WZ_TuneCP5_13TeV-pythia8/MINIAODSIM/106X_mc2017_realistic_v9-v1/280000/BB4FAA03-8C4D-C845-A1A9-D1BE4EAE8820.root -a '-o ./ -m --year 2017'"
transfer_output_remaps="out_1.root = /ospool/cms-user/yuzhe/NtupleStore/Wcb/2017/MC/WZ_TuneCP5_13TeV-pythia8_RunIISummer20UL17MiniAODv2-106X_mc2017_realistic_v9-v1/out_73.root "
queue
        
arguments="-f /store/mc/RunIISummer20UL17MiniAODv2/WZ_TuneCP5_13TeV-pythia8/MINIAODSIM/106X_mc2017_realistic_v9-v1/280000/46CE57B2-7062-F349-986E-EA815EFA37E3.root -a '-o ./ -m --year 2017'"
transfer_output_remaps="out_1.root = /ospool/cms-user/yuzhe/NtupleStore/Wcb/2017/MC/WZ_TuneCP5_13TeV-pythia8_RunIISummer20UL17MiniAODv2-106X_mc2017_realistic_v9-v1/out_72.root "
queue
        
arguments="-f /store/mc/RunIISummer20UL17MiniAODv2/WZ_TuneCP5_13TeV-pythia8/MINIAODSIM/106X_mc2017_realistic_v9-v1/280000/05A715E2-79B9-0D4A-8BD3-F2A51BDFECA3.root -a '-o ./ -m --year 2017'"
transfer_output_remaps="out_1.root = /ospool/cms-user/yuzhe/NtupleStore/Wcb/2017/MC/WZ_TuneCP5_13TeV-pythia8_RunIISummer20UL17MiniAODv2-106X_mc2017_realistic_v9-v1/out_71.root "
queue
        
arguments="-f /store/mc/RunIISummer20UL17MiniAODv2/WZ_TuneCP5_13TeV-pythia8/MINIAODSIM/106X_mc2017_realistic_v9-v1/280000/C94A6337-74B7-9B40-9122-BE5D03D904EB.root -a '-o ./ -m --year 2017'"
transfer_output_remaps="out_1.root = /ospool/cms-user/yuzhe/NtupleStore/Wcb/2017/MC/WZ_TuneCP5_13TeV-pythia8_RunIISummer20UL17MiniAODv2-106X_mc2017_realistic_v9-v1/out_70.root "
queue
        
arguments="-f /store/mc/RunIISummer20UL17MiniAODv2/WZ_TuneCP5_13TeV-pythia8/MINIAODSIM/106X_mc2017_realistic_v9-v1/280000/61CD5519-D9C5-6040-A66C-66D76767F3F5.root -a '-o ./ -m --year 2017'"
transfer_output_remaps="out_1.root = /ospool/cms-user/yuzhe/NtupleStore/Wcb/2017/MC/WZ_TuneCP5_13TeV-pythia8_RunIISummer20UL17MiniAODv2-106X_mc2017_realistic_v9-v1/out_79.root "
queue
        
arguments="-f /store/mc/RunIISummer20UL17MiniAODv2/WZ_TuneCP5_13TeV-pythia8/MINIAODSIM/106X_mc2017_realistic_v9-v1/280000/18176D8A-866B-264D-9418-21E92D9CF25D.root -a '-o ./ -m --year 2017'"
transfer_output_remaps="out_1.root = /ospool/cms-user/yuzhe/NtupleStore/Wcb/2017/MC/WZ_TuneCP5_13TeV-pythia8_RunIISummer20UL17MiniAODv2-106X_mc2017_realistic_v9-v1/out_78.root "
queue
        