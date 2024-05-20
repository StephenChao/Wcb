universe=Vanilla
+ProjectName = "cms.org"
requirements = (OpSysAndVer =?= "CentOS7")
RequestMemory = 2048
RequestCpus = 1
executable=/home/legao/Wcb/condor/production/NanoNtupleChain_25_Apr_2024/SingleMuon_Run2018B-UL2018_MiniAODv2-v2/exe_CUSTNANO_UL18_MINIAODv2_DATA.sh
transfer_executable=True
transfer_input_files=/home/legao/Wcb/condor/x509up_u14433,/home/legao/Wcb/condor/Scripts/fetchFiles.py
transfer_output_files=""
log=/home/legao/Wcb/condor/production/NanoNtupleChain_25_Apr_2024/SingleMuon_Run2018B-UL2018_MiniAODv2-v2/log//$(Cluster).log
output=/home/legao/Wcb/condor/production/NanoNtupleChain_25_Apr_2024/SingleMuon_Run2018B-UL2018_MiniAODv2-v2/log//$(Cluster).out
error=/home/legao/Wcb/condor/production/NanoNtupleChain_25_Apr_2024/SingleMuon_Run2018B-UL2018_MiniAODv2-v2/log//$(Cluster).err
notification=Never
should_transfer_files = YES
when_to_transfer_output = ON_EXIT
x509userproxy=x509up_u14433
+MaxRuntime=150000
on_exit_remove        = (ExitBySignal == False) && (ExitCode == 0)
on_exit_hold          = (ExitBySignal == True) || (ExitCode != 0)
on_exit_hold_reason   = strcat("Job held by ON_EXIT_HOLD due to ", ifThenElse((ExitBySignal == True), strcat("exit signal ", ExitSignal), strcat("exit code ", ExitCode)), ".")
periodic_release      = (NumJobStarts < 3) && ((CurrentTime - EnteredCurrentStatus) > 60*60)
arguments="-f /store/data/Run2018B/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/312F4B7A-5441-CD41-A11B-C549A624C952.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018B-UL2018_MiniAODv2-v2/out_198.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/4D4337C9-0180-6C49-81E7-6F66B6BFB9AE.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018B-UL2018_MiniAODv2-v2/out_344.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/110000/FD4BD608-C1FA-0A4A-A4F9-6633398B00C7.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018B-UL2018_MiniAODv2-v2/out_819.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/891FDBD4-2B6C-F24C-89F8-89D598D0783C.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018B-UL2018_MiniAODv2-v2/out_346.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/4B402B01-8346-D040-8A34-5FD2ACC0B23B.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018B-UL2018_MiniAODv2-v2/out_347.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/577199E0-DDD4-E54A-8108-0138D825C2D9.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018B-UL2018_MiniAODv2-v2/out_340.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/4874C096-F368-BD44-B8F3-58206A15071E.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018B-UL2018_MiniAODv2-v2/out_341.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/E358D127-8DEE-9948-8D8E-214CF57CF027.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018B-UL2018_MiniAODv2-v2/out_342.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/FEB9C05F-F814-BE42-86BC-24FDD6F6F47A.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018B-UL2018_MiniAODv2-v2/out_343.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/110000/103229D2-AD22-D54F-8757-D82FD6EAEC00.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018B-UL2018_MiniAODv2-v2/out_810.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/110000/54BC7178-8616-504A-9172-4B887D451DE2.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018B-UL2018_MiniAODv2-v2/out_811.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/110000/6E0F8369-DBA9-2841-B642-DFD8DE243B16.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018B-UL2018_MiniAODv2-v2/out_812.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/110000/B0DB6E06-B5B7-A443-B448-38FC062D0416.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018B-UL2018_MiniAODv2-v2/out_813.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/BEC337F7-CF0E-AB40-AC2D-16FBC077BF44.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018B-UL2018_MiniAODv2-v2/out_348.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/287CD0BA-4327-FB43-9369-5142439818A3.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018B-UL2018_MiniAODv2-v2/out_349.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/110000/533BE42D-2B41-264B-9419-1F079F77E969.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018B-UL2018_MiniAODv2-v2/out_816.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/110000/8F95C336-9209-2C49-98E8-B539C6DA3CDA.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018B-UL2018_MiniAODv2-v2/out_817.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/110000/52CEB4D8-1FB3-1C42-B0E7-653B111836C5.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018B-UL2018_MiniAODv2-v2/out_737.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/110000/25E4D08C-CF2E-BE41-A4D9-26074BB2F589.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018B-UL2018_MiniAODv2-v2/out_739.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/B9F774B5-C1A8-6F47-9FBE-1260A19E0E99.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018B-UL2018_MiniAODv2-v2/out_719.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/824EADF7-78C5-B749-A4D9-25A05187A1B9.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018B-UL2018_MiniAODv2-v2/out_718.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/489E5DD2-FAE9-A143-8022-01E252F5F6FB.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018B-UL2018_MiniAODv2-v2/out_717.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/F9578367-9ECD-E44D-A20A-9C91E343E313.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018B-UL2018_MiniAODv2-v2/out_716.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/D5CE983C-A2B4-2A42-AD62-27E05276B5AD.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018B-UL2018_MiniAODv2-v2/out_715.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/76C8BE16-F53D-EA48-9FB7-4B6D53886230.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018B-UL2018_MiniAODv2-v2/out_714.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/93F6E004-637B-DD4D-BC1F-071D38F8E13D.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018B-UL2018_MiniAODv2-v2/out_713.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/4BFB061A-75AA-3140-A4A0-1515E5B32E4B.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018B-UL2018_MiniAODv2-v2/out_712.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/6C45043F-AD44-634B-818C-D65202C6C956.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018B-UL2018_MiniAODv2-v2/out_711.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/B9DDF13E-80F0-CC40-B85F-CBC4E537692F.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018B-UL2018_MiniAODv2-v2/out_710.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/76F349A3-236F-0747-9331-1327B9B28432.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018B-UL2018_MiniAODv2-v2/out_421.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/AF74805F-D6E8-CE42-A5BB-E8C370F6F6B9.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018B-UL2018_MiniAODv2-v2/out_129.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/67679E7A-C352-6343-86A2-4EFBDAFA4C6A.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018B-UL2018_MiniAODv2-v2/out_420.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/B59A6FE4-6A62-3E4D-BC89-D44F96DE6A04.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018B-UL2018_MiniAODv2-v2/out_423.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/185610D2-DFE1-3F45-9B58-CCFC89067A76.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018B-UL2018_MiniAODv2-v2/out_1069.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/1047EBB3-EE05-6041-9E8B-6B41D6CF19FD.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018B-UL2018_MiniAODv2-v2/out_1068.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/8111D7F4-8DB2-3945-AA2B-C807D4E4870F.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018B-UL2018_MiniAODv2-v2/out_422.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/42737B16-74E4-0B43-8778-7D161DB59533.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018B-UL2018_MiniAODv2-v2/out_1061.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/06F492DF-89F7-0243-8108-DE0C41B7C92F.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018B-UL2018_MiniAODv2-v2/out_1060.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/674FC569-7F05-904E-8069-DE9655B58FAD.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018B-UL2018_MiniAODv2-v2/out_1063.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/CB60F638-9A87-0542-962F-63397603FA97.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018B-UL2018_MiniAODv2-v2/out_1062.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/3FB2C222-18C9-3D44-99DA-A809FAB260A2.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018B-UL2018_MiniAODv2-v2/out_1065.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/ED1C8E9B-943F-9F42-BB61-766379E0DB52.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018B-UL2018_MiniAODv2-v2/out_1064.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/48EB92F8-3A1F-434A-8585-FFDFE5486411.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018B-UL2018_MiniAODv2-v2/out_1067.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/EBE65B14-F983-9944-8551-634952256B75.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018B-UL2018_MiniAODv2-v2/out_1066.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/C45B3310-C11B-0B44-84A8-33C728359722.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018B-UL2018_MiniAODv2-v2/out_427.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/7EDC1922-B6D5-E749-BD42-9D4F19E4FC27.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018B-UL2018_MiniAODv2-v2/out_298.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/78C732EF-0EB5-6541-9516-C64F1C7CCA75.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018B-UL2018_MiniAODv2-v2/out_299.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/F990B709-FC1D-EE41-8659-9BE1E55F2FB5.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018B-UL2018_MiniAODv2-v2/out_296.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/CA5BE072-975B-0244-B30B-FC61AC0F76C5.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018B-UL2018_MiniAODv2-v2/out_297.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/4102531E-C94C-BB4E-B68D-1F2A5D6AFA3B.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018B-UL2018_MiniAODv2-v2/out_294.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/729F3EA2-3EF6-5446-8AF0-8EAD337EDBB8.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018B-UL2018_MiniAODv2-v2/out_295.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/1C828F72-CCBD-F146-A31D-CDAA1FBAFA12.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018B-UL2018_MiniAODv2-v2/out_292.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/8A3C7A64-7704-8845-BA98-27C4F35814E0.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018B-UL2018_MiniAODv2-v2/out_293.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/7718B232-015B-8841-B979-55880194F6D7.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018B-UL2018_MiniAODv2-v2/out_290.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/20C486E5-3DB4-DD46-9667-964AD97934A0.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018B-UL2018_MiniAODv2-v2/out_291.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/4CAC1DB3-463F-F54B-88B5-B4B556B3A82F.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018B-UL2018_MiniAODv2-v2/out_591.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/A8BE82F1-CE71-7241-A0EA-B68DA6F75043.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018B-UL2018_MiniAODv2-v2/out_590.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/A86B812D-1DD2-B34E-A358-B1F742808625.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018B-UL2018_MiniAODv2-v2/out_1085.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/F064AC52-9204-694C-94D8-324C9C1FD093.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018B-UL2018_MiniAODv2-v2/out_592.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/E714CB32-133C-0D41-A947-19D289105AB0.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018B-UL2018_MiniAODv2-v2/out_595.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/EC153EAD-C2D4-7141-98DF-0E4C190F6CBC.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018B-UL2018_MiniAODv2-v2/out_1082.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/263E30CC-B0E4-624E-9C24-ADAE408F8FD2.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018B-UL2018_MiniAODv2-v2/out_597.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/BC12549E-1A86-0346-B4F1-1C48F89FA085.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018B-UL2018_MiniAODv2-v2/out_596.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/F6F9D03D-184D-A141-A31C-A5EEC83E31CF.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018B-UL2018_MiniAODv2-v2/out_195.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/7C645FAE-69DE-C44D-9941-31288838D2EF.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018B-UL2018_MiniAODv2-v2/out_194.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/A136DA7C-B523-5942-A93F-70A60DDFC8B6.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018B-UL2018_MiniAODv2-v2/out_197.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/CEA58B3F-F8DC-7048-A71C-78F5631B5948.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018B-UL2018_MiniAODv2-v2/out_196.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/41B8F174-4324-0943-9D4D-66BFEBB81821.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018B-UL2018_MiniAODv2-v2/out_191.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/D5E586B3-ADB1-DC4A-842E-51ABAF18B108.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018B-UL2018_MiniAODv2-v2/out_190.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/E144099A-9799-DF48-AA33-C431F68CF621.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018B-UL2018_MiniAODv2-v2/out_193.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/43EC7BFC-6FE2-DF42-924E-B556BC794510.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018B-UL2018_MiniAODv2-v2/out_192.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/08AC3BDA-BA9D-9140-A333-0E824803F1CC.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018B-UL2018_MiniAODv2-v2/out_270.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/BF35481A-63F7-2F4F-9A64-1C70715C965B.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018B-UL2018_MiniAODv2-v2/out_271.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/C63C842F-9131-2B49-A37F-B7CBE34FE497.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018B-UL2018_MiniAODv2-v2/out_272.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70002/3A62A9F0-09AF-8A41-89B0-2D5C3E0F184E.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018B-UL2018_MiniAODv2-v2/out_273.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/05229AEF-6530-974D-BF71-0B84F2CAB35D.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018B-UL2018_MiniAODv2-v2/out_274.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/7CBFF46F-0311-C841-B4AF-A29B748E2D2C.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018B-UL2018_MiniAODv2-v2/out_275.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/50559414-AE13-2749-B265-68F6457BDCA5.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018B-UL2018_MiniAODv2-v2/out_276.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/C4257751-194A-2D4D-A76E-3100F2A01C68.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018B-UL2018_MiniAODv2-v2/out_277.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/BF78B075-C78E-9E47-8EAC-03027D9F51A4.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018B-UL2018_MiniAODv2-v2/out_278.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/19E8D101-4BAB-D245-9718-4519C8B21FB2.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018B-UL2018_MiniAODv2-v2/out_279.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/94C000BA-A1EB-874D-AF9E-5886B9CD51C9.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018B-UL2018_MiniAODv2-v2/out_569.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/14B35C32-EA81-DF46-ABBF-AB6816C84D21.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018B-UL2018_MiniAODv2-v2/out_497.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/BAC7044D-B317-224A-88E9-13163F487678.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018B-UL2018_MiniAODv2-v2/out_524.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/9C23F969-96AE-E64D-B54B-F9151B474F78.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018B-UL2018_MiniAODv2-v2/out_525.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/5B6BF2E2-EBD3-954A-A0F0-8B1FB71CA4E4.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018B-UL2018_MiniAODv2-v2/out_526.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/BFA31407-5216-2742-9A11-5F0E04D6D6FB.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018B-UL2018_MiniAODv2-v2/out_527.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/7B63CBD7-4CBD-4C47-9DA1-53476D5DCC27.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018B-UL2018_MiniAODv2-v2/out_520.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/0277A9DD-75F9-0D4F-A091-38906831724E.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018B-UL2018_MiniAODv2-v2/out_521.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/1FE67EA9-ECEF-CE4C-A844-171322A13A95.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018B-UL2018_MiniAODv2-v2/out_522.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/F4713ABA-18D7-F343-9DBC-D278D84D2C3A.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018B-UL2018_MiniAODv2-v2/out_523.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/110000/98CEA592-BD1C-FD4A-ABD7-2E8A041F3E05.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018B-UL2018_MiniAODv2-v2/out_1014.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/110000/50E548C4-C06F-834D-A01E-14039B72C8BA.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018B-UL2018_MiniAODv2-v2/out_1015.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/110000/65186B26-EBA9-DA4B-8F37-96EC27F5FBAE.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018B-UL2018_MiniAODv2-v2/out_1016.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/110000/592B575A-EE70-184C-A8EF-F88F05EB9FC7.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018B-UL2018_MiniAODv2-v2/out_1017.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/7100DBEB-A436-134D-93FD-D281EA68BF15.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018B-UL2018_MiniAODv2-v2/out_528.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/A81C6863-3B26-2F42-A772-DCC2A0122CF7.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018B-UL2018_MiniAODv2-v2/out_529.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/110000/CD15852B-3C05-4B41-B75A-45E6636E1D76.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018B-UL2018_MiniAODv2-v2/out_1012.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/110000/38350061-DC74-DC4B-8BCF-3CFEB0335365.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018B-UL2018_MiniAODv2-v2/out_1013.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/F2FBAD73-D6BD-7D42-9A70-062765E0631C.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018B-UL2018_MiniAODv2-v2/out_449.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/99CBC25B-6BA7-3E48-8EC3-22C7E71BA8CB.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018B-UL2018_MiniAODv2-v2/out_448.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/F48CB3F5-7844-9D47-BDD9-B51802545324.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018B-UL2018_MiniAODv2-v2/out_345.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/96180813-2C8A-D442-95B0-1535E45EC23C.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018B-UL2018_MiniAODv2-v2/out_443.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/0F939DE1-83A2-4E4D-BD9B-2735109C5738.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018B-UL2018_MiniAODv2-v2/out_442.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/9BB274CC-54E8-A546-8438-64B55B1E445B.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018B-UL2018_MiniAODv2-v2/out_441.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/95D3F76C-7031-6145-AF49-FE1D34EF9412.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018B-UL2018_MiniAODv2-v2/out_440.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/AC033B09-3706-4646-A5F5-AEA704CF0BA8.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018B-UL2018_MiniAODv2-v2/out_447.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/21E9A4DC-EFF4-084B-ADBE-6BEE31F942D4.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018B-UL2018_MiniAODv2-v2/out_446.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/0FDAE976-DA99-1C44-8DDF-2A663C0409C7.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018B-UL2018_MiniAODv2-v2/out_445.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/204147BF-C895-7848-8A61-0CC5486402FD.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018B-UL2018_MiniAODv2-v2/out_444.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/260000/60793A43-879C-844F-85AF-134EB1872E5B.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018B-UL2018_MiniAODv2-v2/out_47.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/260000/D438B938-A331-0645-A608-777BBE1587B2.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018B-UL2018_MiniAODv2-v2/out_108.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/260000/14C9F1A3-D2A6-B44F-9CFA-621BF74E5461.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018B-UL2018_MiniAODv2-v2/out_109.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/260000/CC033DE4-D35F-3441-987C-57DB2878D1E7.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018B-UL2018_MiniAODv2-v2/out_102.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/260000/470C9F6F-31BD-8745-99E0-CDE73A0A1BA5.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018B-UL2018_MiniAODv2-v2/out_103.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/260000/9DC8C744-05AC-3C4C-A2A8-D854C1A3DD9C.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018B-UL2018_MiniAODv2-v2/out_100.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/260000/B402326B-D22E-E841-B31D-EE3DB5205AB2.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018B-UL2018_MiniAODv2-v2/out_101.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/260000/0B3397E9-116F-F14F-B936-E543A55426C3.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018B-UL2018_MiniAODv2-v2/out_106.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/260000/B5392D52-1712-9B47-81F2-B8873F056F67.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018B-UL2018_MiniAODv2-v2/out_107.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/260000/1BF3CE64-C1B7-B748-9ECD-3E93639B20A0.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018B-UL2018_MiniAODv2-v2/out_104.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/260000/40350A7E-E43C-BC4C-AA00-99A93887F56D.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018B-UL2018_MiniAODv2-v2/out_105.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/110000/CCD571FC-16C4-7E41-87E9-1B5ABDAB04F1.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018B-UL2018_MiniAODv2-v2/out_902.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/110000/5EE70ADA-F5CE-B245-BD12-3EE7AF9AC100.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018B-UL2018_MiniAODv2-v2/out_903.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/260000/0211A354-CEC2-2F4D-B5EE-B2658EF2A3F2.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018B-UL2018_MiniAODv2-v2/out_39.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/260000/FC1458E8-089F-0649-AC84-9B5695451C8A.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018B-UL2018_MiniAODv2-v2/out_38.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/110000/96C3ECE2-774A-C546-8404-A4CE2D286E98.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018B-UL2018_MiniAODv2-v2/out_906.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/110000/1E9CB6FC-2419-BF48-B07B-3C21A9CCE44B.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018B-UL2018_MiniAODv2-v2/out_907.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/110000/D3C6E116-8A57-1043-AEC2-232700605412.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018B-UL2018_MiniAODv2-v2/out_904.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/110000/CD1CCE82-0AF9-2443-A171-68F3899BA0DE.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018B-UL2018_MiniAODv2-v2/out_905.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/260000/1320A774-DE43-E247-AC2D-81B37863E7B0.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018B-UL2018_MiniAODv2-v2/out_33.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/260000/BADC0DAF-6142-0F4F-98A8-65CDAACF005F.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018B-UL2018_MiniAODv2-v2/out_32.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/260000/9145E1AC-770E-E940-8665-AD75A2352450.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018B-UL2018_MiniAODv2-v2/out_31.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/260000/A54FD7F3-68C0-404D-A0EF-6AAB1CE652C9.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018B-UL2018_MiniAODv2-v2/out_30.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/260000/A5AB24DC-4C02-5C40-8224-06E4749C6A80.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018B-UL2018_MiniAODv2-v2/out_37.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/260000/34B054B3-9730-BB45-88DA-2A57D6285B99.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018B-UL2018_MiniAODv2-v2/out_36.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/260000/23874768-A136-E945-9A3A-9092DB5582E8.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018B-UL2018_MiniAODv2-v2/out_35.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/260000/CEE52BF1-0522-B247-8845-46C1B9C2A2EA.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018B-UL2018_MiniAODv2-v2/out_34.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/09D2A942-878F-9049-AD9C-B67402C33ADA.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018B-UL2018_MiniAODv2-v2/out_641.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/E3C824B6-848F-E841-B515-558954CED3AE.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018B-UL2018_MiniAODv2-v2/out_640.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/8A4D2BA0-9916-9A41-B28E-44C102052647.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018B-UL2018_MiniAODv2-v2/out_643.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/22B1EA4E-4279-6548-B00B-FA7D55F90D3B.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018B-UL2018_MiniAODv2-v2/out_642.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/1A820D0F-C608-A64A-B4A1-1C00A1D0236C.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018B-UL2018_MiniAODv2-v2/out_645.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/B30383FA-975D-F247-996E-62274F538FE3.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018B-UL2018_MiniAODv2-v2/out_644.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/FF4DFD6D-5DAC-2743-96D7-309ECC5A2FC9.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018B-UL2018_MiniAODv2-v2/out_438.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/AA261293-1758-324F-8A07-4EDEA9B3CD21.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018B-UL2018_MiniAODv2-v2/out_439.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/3AEECE0A-9666-0446-A985-A6041B697545.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018B-UL2018_MiniAODv2-v2/out_436.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/168C35E3-4B51-8344-B53C-DCCD61399EA1.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018B-UL2018_MiniAODv2-v2/out_437.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/E040A35F-393B-7C4B-8F03-69DCD02039E8.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018B-UL2018_MiniAODv2-v2/out_434.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/F0950263-F811-7D4A-9927-75D651533FD4.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018B-UL2018_MiniAODv2-v2/out_435.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/E8DFACA7-1FA5-B843-82E6-CFDEA4B9455D.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018B-UL2018_MiniAODv2-v2/out_432.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/D8229EEF-0FAD-554D-B0E2-0A044F342C5F.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018B-UL2018_MiniAODv2-v2/out_433.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/EC6547B8-0D83-5044-891A-6F96E8355867.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018B-UL2018_MiniAODv2-v2/out_430.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/43CABA99-20BC-4049-B860-64FBE5AA0935.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018B-UL2018_MiniAODv2-v2/out_431.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/110000/B919060C-5D57-4541-9980-5FC776CFADE8.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018B-UL2018_MiniAODv2-v2/out_826.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/9DCB3C97-87F5-DC4C-8603-994B881A3AE3.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018B-UL2018_MiniAODv2-v2/out_339.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/60E6CBE1-91ED-C047-BB6F-C7981F0E2B76.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018B-UL2018_MiniAODv2-v2/out_338.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/2131202F-3A64-A645-8A54-EE3598E8A98C.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018B-UL2018_MiniAODv2-v2/out_335.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/4348A1B3-D065-4C4E-9BCB-13ED2D891727.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018B-UL2018_MiniAODv2-v2/out_334.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/696B37D5-2116-1142-8D16-177F448A62FD.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018B-UL2018_MiniAODv2-v2/out_337.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/D2D6AB8E-AC8C-C845-89B8-D554B6F73831.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018B-UL2018_MiniAODv2-v2/out_336.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/42F4B379-55BA-5245-B8EE-45A1577EDD5C.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018B-UL2018_MiniAODv2-v2/out_331.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/BAFA1D50-52D2-C24B-A4B4-C00F1BA1F6DE.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018B-UL2018_MiniAODv2-v2/out_330.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/60BB1128-A2B2-A945-BDB8-E379B88A62E8.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018B-UL2018_MiniAODv2-v2/out_333.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/1EDEF12B-8836-B448-B33A-6101EE29F89F.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018B-UL2018_MiniAODv2-v2/out_332.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/755D42EE-F4A9-2649-99BD-BA5A03E44F50.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018B-UL2018_MiniAODv2-v2/out_559.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/110000/8BAD8AF0-83AE-D647-A8BD-E4ED1213DEA3.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018B-UL2018_MiniAODv2-v2/out_1026.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/110000/8DE23D1B-0AA0-D94F-B0D6-58A3F0F6233C.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018B-UL2018_MiniAODv2-v2/out_745.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/110000/285E76FA-9F1C-214F-8F8B-C4FFFCFBC6C0.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018B-UL2018_MiniAODv2-v2/out_854.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/110000/4F48BE32-5387-D044-98BF-2E8C1A515F72.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018B-UL2018_MiniAODv2-v2/out_818.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/3E784138-572C-C240-B23A-824F8377D384.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018B-UL2018_MiniAODv2-v2/out_856.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/4E850737-4283-6246-9C11-403FF893448B.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018B-UL2018_MiniAODv2-v2/out_857.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/110000/3F6CD71D-6D56-9647-8C90-AEBFCF8FEB9C.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018B-UL2018_MiniAODv2-v2/out_850.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/110000/3B7B92A7-E149-E64A-A226-4A0A39FA171D.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018B-UL2018_MiniAODv2-v2/out_851.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/110000/31F6CAE1-4C9D-104F-BC48-754D3C754BFB.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018B-UL2018_MiniAODv2-v2/out_852.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/260000/DE08786D-96A0-0544-9C84-CC53630BA477.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018B-UL2018_MiniAODv2-v2/out_0.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/7F12A724-DDBF-514D-835F-4E1923FB4B00.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018B-UL2018_MiniAODv2-v2/out_858.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/61337802-DFB9-5446-AC85-2522CA80DDB1.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018B-UL2018_MiniAODv2-v2/out_859.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/110000/A52FDF22-99BB-0348-B642-F52D60947CC9.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018B-UL2018_MiniAODv2-v2/out_748.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/260000/D60E7032-C334-3449-AE42-6C3E64FD9AF7.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018B-UL2018_MiniAODv2-v2/out_6.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/13F1A426-E7AB-694C-B686-50E61E39CB3E.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018B-UL2018_MiniAODv2-v2/out_425.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/260000/2C80AB56-2C88-5A48-BA1D-FD97A5EBE354.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018B-UL2018_MiniAODv2-v2/out_43.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/260000/E14C1299-698C-904A-8F66-0BB9C2B20F14.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018B-UL2018_MiniAODv2-v2/out_99.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/260000/16771368-2295-6049-AB8B-11180B2CE426.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018B-UL2018_MiniAODv2-v2/out_98.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/260000/D977CBE3-EFFC-EE4B-A648-E5B59E5C52BC.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018B-UL2018_MiniAODv2-v2/out_8.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/260000/A4952552-E962-1E43-BCB9-09D1C95D62AC.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018B-UL2018_MiniAODv2-v2/out_91.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/260000/DCBA58EF-55BC-F740-9D2D-09B7FE389FE8.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018B-UL2018_MiniAODv2-v2/out_90.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/260000/19BCAE8C-456A-2F41-93D9-7DC6D365EC75.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018B-UL2018_MiniAODv2-v2/out_93.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/260000/8D12F5D5-A332-CE43-B589-FA4F22D4D50F.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018B-UL2018_MiniAODv2-v2/out_92.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/260000/B0554E18-779A-734E-A570-EBDD185BE0FD.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018B-UL2018_MiniAODv2-v2/out_95.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/260000/63920341-F22D-3D46-8816-733E5EEF7BE7.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018B-UL2018_MiniAODv2-v2/out_94.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/260000/AA02EB3A-B03D-3247-B6F8-48F5CF9C17BC.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018B-UL2018_MiniAODv2-v2/out_97.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/260000/969801DF-0AB7-7846-9E7C-E1D002488061.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018B-UL2018_MiniAODv2-v2/out_96.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/2D416042-68A4-6447-86F1-A58B3B2D973A.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018B-UL2018_MiniAODv2-v2/out_424.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/110000/C48DD8B7-F603-CB49-A4B2-331183A881D7.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018B-UL2018_MiniAODv2-v2/out_814.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/110000/B955914F-91C1-3D4E-BFD7-990D7E0A7A19.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018B-UL2018_MiniAODv2-v2/out_815.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/418FB10D-13C9-1547-9B2F-4A8B63457D4E.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018B-UL2018_MiniAODv2-v2/out_153.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/110000/0EECC76E-E1EC-F748-B859-74B0A7E447FA.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018B-UL2018_MiniAODv2-v2/out_740.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/110000/F9E0130E-9D50-954F-B2B6-D4C80E69217C.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018B-UL2018_MiniAODv2-v2/out_741.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/110000/5CA62B5B-B052-394C-87DD-F3EDFDDA0964.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018B-UL2018_MiniAODv2-v2/out_742.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/110000/2B125863-BFB2-854B-8367-913C8F9AB768.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018B-UL2018_MiniAODv2-v2/out_743.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/110000/A9E64A91-9623-FD4D-976D-8E3D9F31A806.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018B-UL2018_MiniAODv2-v2/out_744.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/DC29C68E-6D3C-8947-BFA3-9DE9AA0A3AC8.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018B-UL2018_MiniAODv2-v2/out_558.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/110000/C90D1EC1-BCB9-8D49-9826-E7BB7DEEF072.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018B-UL2018_MiniAODv2-v2/out_746.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/110000/ABA030FF-11C1-BC46-BBC0-C0E756951F63.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018B-UL2018_MiniAODv2-v2/out_747.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/334BB648-1271-1240-A083-0C86AD519F14.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018B-UL2018_MiniAODv2-v2/out_555.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/A309D158-D8B3-6A47-AED3-EE78B35D2F7E.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018B-UL2018_MiniAODv2-v2/out_554.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/1DEF2A93-5958-2642-932E-2EBC0E97F990.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018B-UL2018_MiniAODv2-v2/out_557.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/E31EF9E3-3F45-3C4D-B450-CEE8A926B7E7.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018B-UL2018_MiniAODv2-v2/out_556.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/747AB33C-7001-7049-85E8-2CFDE5854D82.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018B-UL2018_MiniAODv2-v2/out_551.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/2ABF9881-203A-7547-8D17-80E9AEC30FE2.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018B-UL2018_MiniAODv2-v2/out_550.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/8B946B54-0D03-8F42-A358-049926B2D638.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018B-UL2018_MiniAODv2-v2/out_553.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/245C99E4-568A-344A-8F4F-C230EBBB6244.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018B-UL2018_MiniAODv2-v2/out_552.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/9E276A42-10DF-514B-8B7F-8F49743D5A94.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018B-UL2018_MiniAODv2-v2/out_238.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/3D09434F-9030-E343-9A32-6DEACA68E8D1.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018B-UL2018_MiniAODv2-v2/out_239.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/662606A5-2206-924D-9F9E-03A11B108917.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018B-UL2018_MiniAODv2-v2/out_234.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/3451144B-9B81-554B-BFD8-D53A03BBF51F.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018B-UL2018_MiniAODv2-v2/out_235.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/6A8CE002-D687-4540-837E-09236CB3C9F7.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018B-UL2018_MiniAODv2-v2/out_236.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/8FB102CB-FC75-944F-9583-21CDA8A24F2E.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018B-UL2018_MiniAODv2-v2/out_237.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/A8F5E531-6292-D444-853E-7F7676A74B85.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018B-UL2018_MiniAODv2-v2/out_230.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/6CB0833B-2264-B646-BC00-B231A0947670.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018B-UL2018_MiniAODv2-v2/out_231.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/3909AEAD-9641-B141-805B-515BC1775BC5.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018B-UL2018_MiniAODv2-v2/out_232.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/14AE66CD-FB1E-0743-9C9F-8EEAE08283B9.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018B-UL2018_MiniAODv2-v2/out_233.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/110000/27B323C6-0492-EA47-946C-7F36984E4AA5.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018B-UL2018_MiniAODv2-v2/out_1050.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/110000/7ABB0F6C-10FA-1340-B7E2-A757BCECC9A5.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018B-UL2018_MiniAODv2-v2/out_1051.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/98A0E9D6-9213-2640-ADFE-2E0BEC38D365.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018B-UL2018_MiniAODv2-v2/out_1052.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/5F87CEDB-5088-9142-9497-75959353B180.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018B-UL2018_MiniAODv2-v2/out_1053.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/3DB596CF-1351-FF48-96A7-9436C1325249.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018B-UL2018_MiniAODv2-v2/out_1054.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/37C84CB9-8776-1846-BE3B-9EB852CE1473.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018B-UL2018_MiniAODv2-v2/out_1055.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/079BD481-6300-3848-A860-FA2E05FA1374.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018B-UL2018_MiniAODv2-v2/out_1056.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/24AEAD4A-5AB1-6D41-9D7A-63A4DDA77815.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018B-UL2018_MiniAODv2-v2/out_1057.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/3AA173E6-3E6A-D349-B4EC-498114339146.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018B-UL2018_MiniAODv2-v2/out_1058.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/5960114D-9DD8-FA4C-9CAD-5356C2E866C3.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018B-UL2018_MiniAODv2-v2/out_1059.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/260000/C4FBEFD6-6D49-D04F-ABBD-E700EF6CFC38.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018B-UL2018_MiniAODv2-v2/out_1.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/3099C451-A2EF-2D42-9686-5242B4E11665.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018B-UL2018_MiniAODv2-v2/out_155.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/3C52251F-F1E8-004A-B9F0-780C4BAEE231.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018B-UL2018_MiniAODv2-v2/out_146.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/33621D26-FE0F-9A45-8107-1E1ABB746ED0.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018B-UL2018_MiniAODv2-v2/out_147.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/D0DFFB26-D2C3-364E-84CC-3114B5354F37.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018B-UL2018_MiniAODv2-v2/out_144.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/4BD5366C-DB2E-5D4B-AE0A-FB72C1F0CDF3.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018B-UL2018_MiniAODv2-v2/out_145.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/D523CD21-C3FE-BC43-A8E3-EEE21BB0F1CC.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018B-UL2018_MiniAODv2-v2/out_142.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/5B6CF2D5-6505-AB4C-B0ED-7E83C88D8D70.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018B-UL2018_MiniAODv2-v2/out_143.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/95C2E70D-7A7D-FD49-AEA5-6453905375F9.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018B-UL2018_MiniAODv2-v2/out_140.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/4CF47D30-658B-A342-AF24-801FF60682F0.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018B-UL2018_MiniAODv2-v2/out_141.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/BFCF91CB-96C8-3F4F-9D37-82429A480302.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018B-UL2018_MiniAODv2-v2/out_612.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/853D61F3-3F67-AE43-98A9-321C67161D4F.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018B-UL2018_MiniAODv2-v2/out_613.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/C29FDE80-CD52-9446-BFC8-539F65E25F2E.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018B-UL2018_MiniAODv2-v2/out_610.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/E502F690-FD80-494B-BF64-F2A1C179C5F8.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018B-UL2018_MiniAODv2-v2/out_611.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/170ABD6A-20AC-2A45-A255-5AFCB8815522.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018B-UL2018_MiniAODv2-v2/out_616.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/73EBD608-C418-4E48-A28B-D2663B43F3A6.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018B-UL2018_MiniAODv2-v2/out_617.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/1DFE56DD-A6EC-C140-BDBE-E5E7DCDD9F64.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018B-UL2018_MiniAODv2-v2/out_148.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/8873D27E-F8E2-F94B-B1AC-01D418D32D26.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018B-UL2018_MiniAODv2-v2/out_149.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/8C700D0B-6C1D-8749-AC2D-023FD0341579.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018B-UL2018_MiniAODv2-v2/out_511.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/E658D3AA-4790-3A42-A368-0B6A975FBD39.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018B-UL2018_MiniAODv2-v2/out_510.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/6A38A738-9046-414C-9F6A-03EACB986302.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018B-UL2018_MiniAODv2-v2/out_154.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/33B5B9F2-1836-0546-9DD5-396B6EF82E7E.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018B-UL2018_MiniAODv2-v2/out_513.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/726E63C8-EE3C-1C49-8D1C-F463AE9BDA54.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018B-UL2018_MiniAODv2-v2/out_512.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/110000/BF1E606D-7B1B-EC47-ACB0-F800C8668E94.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018B-UL2018_MiniAODv2-v2/out_948.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/110000/8B2759D9-AD5C-5B4F-B1F4-F7F19A54F0E8.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018B-UL2018_MiniAODv2-v2/out_949.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/110000/F33B0B15-D78E-2742-92CF-1E1A79BECFF2.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018B-UL2018_MiniAODv2-v2/out_946.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/110000/E478A623-2F29-2C48-B7E1-D0783C5BF572.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018B-UL2018_MiniAODv2-v2/out_947.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/110000/C9DF0D73-4450-DD4C-8CED-A6B2E29548B3.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018B-UL2018_MiniAODv2-v2/out_944.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/110000/A303F0FF-E95D-9B42-895F-D33100B5A70C.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018B-UL2018_MiniAODv2-v2/out_945.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/110000/4CC6D45A-B07F-4445-9D69-4CD9B52F89AE.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018B-UL2018_MiniAODv2-v2/out_942.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/110000/C7C6623D-E73D-B348-ADD1-705A84C45DA5.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018B-UL2018_MiniAODv2-v2/out_943.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/110000/F4ED4938-4924-6A40-AFB5-34170E49FBFE.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018B-UL2018_MiniAODv2-v2/out_940.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/110000/F91FFB83-DFC6-E049-AA14-2DE57FA968B1.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018B-UL2018_MiniAODv2-v2/out_941.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/FE3BE0B9-95DA-2448-BC55-B8CEC0FB0B08.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018B-UL2018_MiniAODv2-v2/out_689.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/29B48994-B17D-B34F-B21E-BE102C4AB74B.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018B-UL2018_MiniAODv2-v2/out_688.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/9AC1B849-7CE6-EF40-8FEC-DDE2479D61D0.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018B-UL2018_MiniAODv2-v2/out_685.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/67AAB349-65B5-1E4F-A353-37810B75E590.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018B-UL2018_MiniAODv2-v2/out_684.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/C459E835-D1D9-984B-8274-A34F3450B3A3.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018B-UL2018_MiniAODv2-v2/out_687.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/91043FA7-8B41-FA41-9614-B2539B77023E.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018B-UL2018_MiniAODv2-v2/out_686.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/3D0C0DAE-F39E-FA4B-BFA4-73C64FA80C64.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018B-UL2018_MiniAODv2-v2/out_681.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/06786FE8-6EC1-C347-8F4A-8F8AB3C7C45D.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018B-UL2018_MiniAODv2-v2/out_680.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/C42035E6-A692-1247-B6D3-ADBB554D8FB9.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018B-UL2018_MiniAODv2-v2/out_683.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/73423FE7-1A2A-254B-8FC1-EC8B41BDA364.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018B-UL2018_MiniAODv2-v2/out_682.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/9A6700A0-1335-5949-924D-C2E75598BEB5.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018B-UL2018_MiniAODv2-v2/out_458.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/3CB9529D-7C46-9D46-A7C9-6B39CE075B07.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018B-UL2018_MiniAODv2-v2/out_459.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/6AAF17E6-8513-AB4B-B52E-8E6EFE1A9E4C.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018B-UL2018_MiniAODv2-v2/out_133.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/B5B765E7-6CC0-224D-B3EB-C040E7A25E46.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018B-UL2018_MiniAODv2-v2/out_132.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/7636EABD-4D69-E441-8E88-4709413C0851.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018B-UL2018_MiniAODv2-v2/out_131.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/57F38997-6017-034C-B33B-FF1543813FCF.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018B-UL2018_MiniAODv2-v2/out_130.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/6BDDCA96-3E5C-4247-A0D2-9E078C304D6F.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018B-UL2018_MiniAODv2-v2/out_137.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/E6E121CE-C365-B348-B25C-B0EDC22128D3.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018B-UL2018_MiniAODv2-v2/out_136.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/F5525A3E-000D-464C-938C-56540A1EDF5F.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018B-UL2018_MiniAODv2-v2/out_135.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/83DD333E-6F64-7145-BBFA-8ADACFD3DCA0.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018B-UL2018_MiniAODv2-v2/out_134.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/35D8055E-6905-B941-9766-F4B418812A3C.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018B-UL2018_MiniAODv2-v2/out_494.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/84E58BD0-7317-6445-9F41-E160A1BFD0C4.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018B-UL2018_MiniAODv2-v2/out_495.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/B6575D02-6061-E944-9D08-CFC9FE2F0251.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018B-UL2018_MiniAODv2-v2/out_139.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/C6D66980-4778-9941-BFAA-F023B93F5D0A.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018B-UL2018_MiniAODv2-v2/out_138.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/B1B09CC5-62F3-D441-8000-D5F7ACA7FB59.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018B-UL2018_MiniAODv2-v2/out_490.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/ABA2211F-FC0B-1340-954B-841CA1C75D52.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018B-UL2018_MiniAODv2-v2/out_491.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/9560A57C-ACCA-7E4D-955E-8A8DC41A8CF3.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018B-UL2018_MiniAODv2-v2/out_492.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/19F542DB-4BD9-B240-9DB1-74359CE0DD90.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018B-UL2018_MiniAODv2-v2/out_493.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/260000/26E196F6-5BFA-7941-A6ED-B6D61638300E.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018B-UL2018_MiniAODv2-v2/out_24.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/260000/CB2534F1-AB5C-E343-861C-3525AED5717E.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018B-UL2018_MiniAODv2-v2/out_25.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/260000/D27D2C23-2677-5547-9DE8-58C179A2B5DD.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018B-UL2018_MiniAODv2-v2/out_26.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/260000/8C0CBE28-4D3C-B94A-A2FD-F218DBAFFBF4.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018B-UL2018_MiniAODv2-v2/out_27.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/260000/B83EEABB-B4A6-A34A-9A78-8D6091345103.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018B-UL2018_MiniAODv2-v2/out_20.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/260000/553C7F83-B4DD-2248-BDD9-E549DA3C7AA2.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018B-UL2018_MiniAODv2-v2/out_21.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/260000/AEF64FF6-2CF2-154D-9556-054067A42AFC.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018B-UL2018_MiniAODv2-v2/out_22.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/260000/8DE0BFEB-8FD5-9F4E-B1D3-BD66297AA17F.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018B-UL2018_MiniAODv2-v2/out_23.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/260000/7740EE12-F6F9-5B4F-941B-68FDA5618FA6.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018B-UL2018_MiniAODv2-v2/out_28.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/260000/52EB24E8-E12E-1243-B8D2-B9263F3D5413.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018B-UL2018_MiniAODv2-v2/out_29.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/110000/00A4B13C-EBCD-6C46-93D7-FA721EA4DB41.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018B-UL2018_MiniAODv2-v2/out_820.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/DA551E40-2F90-F147-BA02-778703E7600A.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018B-UL2018_MiniAODv2-v2/out_407.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/731D0FD9-B8D3-984A-A3F1-0864E8D77489.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018B-UL2018_MiniAODv2-v2/out_406.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/6BAB6365-7167-E04C-B6F4-CA4B28723852.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018B-UL2018_MiniAODv2-v2/out_405.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/20EA3C98-E8A1-7748-95E0-61F6458C1D8A.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018B-UL2018_MiniAODv2-v2/out_404.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/A9E278FE-301D-4944-9CCD-AD41E63BCEE8.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018B-UL2018_MiniAODv2-v2/out_403.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/499ECB37-F24C-5F49-8CDF-DEED377AEFD9.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018B-UL2018_MiniAODv2-v2/out_402.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/C453D2ED-7E5E-7F4B-9EC9-20FF8CAC4E0E.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018B-UL2018_MiniAODv2-v2/out_401.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/0E970B62-6D4A-7E4F-9366-39A334741182.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018B-UL2018_MiniAODv2-v2/out_400.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/110000/A229C2CB-3B3A-4844-8BB6-28598CEDAD67.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018B-UL2018_MiniAODv2-v2/out_933.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/110000/98E521F4-8A8B-AE43-A855-05334322CA1F.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018B-UL2018_MiniAODv2-v2/out_932.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/110000/B3450C8B-5A06-A84A-B04F-BE7450B9B7D3.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018B-UL2018_MiniAODv2-v2/out_931.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/110000/F16DE3FF-88A1-C840-9DA2-DDBFF2132C42.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018B-UL2018_MiniAODv2-v2/out_930.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/110000/B151C868-41F0-044E-8F1E-587380096708.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018B-UL2018_MiniAODv2-v2/out_937.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/AB1A00B6-A0A4-484B-8048-2A5CE58EEF69.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018B-UL2018_MiniAODv2-v2/out_452.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/1052F3B6-7365-AF41-B76A-DCDB1C57196A.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018B-UL2018_MiniAODv2-v2/out_409.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/C57D6B23-1436-5246-AA3D-C9640A7293A8.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018B-UL2018_MiniAODv2-v2/out_408.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/407EB824-15AA-9A4A-897E-4D3EC437373E.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018B-UL2018_MiniAODv2-v2/out_453.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/77D16ACF-6E9F-5749-803B-FB3AE9EC2BDF.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018B-UL2018_MiniAODv2-v2/out_414.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/0C4DC80E-8A37-1648-8E7C-B9901424C244.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018B-UL2018_MiniAODv2-v2/out_455.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/270000/2EF8EFE1-67E9-8B4C-8B22-FD8096F8B054.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018B-UL2018_MiniAODv2-v2/out_183.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/2DA53B38-BE48-0F48-AB2A-B5CA29E9DD47.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018B-UL2018_MiniAODv2-v2/out_415.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/C23D6C55-8538-A64C-82C8-15EB77B58223.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018B-UL2018_MiniAODv2-v2/out_379.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/7781E9FD-CFC7-4149-B9AF-83869731AEDA.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018B-UL2018_MiniAODv2-v2/out_378.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/0864E955-D728-3145-BB9F-874FEDDCD10D.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018B-UL2018_MiniAODv2-v2/out_228.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/110000/F51D159E-609C-3A4C-9DC7-2E7B1CC1A0B0.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018B-UL2018_MiniAODv2-v2/out_829.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/110000/0AF438E3-4BD0-2C45-94DE-DB70E3A673D9.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018B-UL2018_MiniAODv2-v2/out_828.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/8C330B22-C2F6-D34A-A9F5-A6C8731D97BF.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018B-UL2018_MiniAODv2-v2/out_371.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/D0690FFC-762F-B546-A081-87637E114AD9.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018B-UL2018_MiniAODv2-v2/out_370.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/2EADD025-BA2A-DD47-B3A7-86EB3189B21A.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018B-UL2018_MiniAODv2-v2/out_373.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/FB92E9E4-74A1-CE40-8443-E8C0B8FC4B6F.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018B-UL2018_MiniAODv2-v2/out_372.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/51A0E1AA-A1C9-8C4F-AE66-35EC406EADAA.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018B-UL2018_MiniAODv2-v2/out_375.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/695F3A2B-EDDE-3D48-B2AD-75384A371DFE.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018B-UL2018_MiniAODv2-v2/out_374.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/4DF3C0F3-A7DA-8046-B48A-54366F1504F4.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018B-UL2018_MiniAODv2-v2/out_377.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/34BAA12E-908B-3044-9D38-23818235EBB8.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018B-UL2018_MiniAODv2-v2/out_376.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/5666A5AC-9E33-D248-8B06-908171B00CD6.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018B-UL2018_MiniAODv2-v2/out_1083.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/99031295-CEAA-894A-85AC-7562F391A560.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018B-UL2018_MiniAODv2-v2/out_708.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/21911DE5-E8AC-B548-8515-56636B2902C6.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018B-UL2018_MiniAODv2-v2/out_709.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/6E96F7FF-1529-6240-A9A9-55FA9E19862F.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018B-UL2018_MiniAODv2-v2/out_704.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/B969B65C-1A4E-E24C-B0C0-2D4AF5E29143.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018B-UL2018_MiniAODv2-v2/out_705.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/E15BCF72-4D02-5745-B03B-3F44296FEE73.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018B-UL2018_MiniAODv2-v2/out_706.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/79B01D82-AE8E-4343-BC0B-5D11F96321D7.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018B-UL2018_MiniAODv2-v2/out_707.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/9AFD62BE-F27C-8843-9E1C-8805B5CCE49D.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018B-UL2018_MiniAODv2-v2/out_700.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/D1E97360-442D-7C48-9865-F760ECBC595C.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018B-UL2018_MiniAODv2-v2/out_618.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/E6486116-0265-A249-ABBA-D83807D2E9BE.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018B-UL2018_MiniAODv2-v2/out_702.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/BDF16129-B5CF-FB4C-94D4-AC0C0B62DECD.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018B-UL2018_MiniAODv2-v2/out_703.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/9D914150-DF72-794B-8C57-AFF0DB80C025.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018B-UL2018_MiniAODv2-v2/out_393.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/C35EB6AF-4183-7445-A100-C221B4023572.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018B-UL2018_MiniAODv2-v2/out_392.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/260000/528522A3-743A-3A4D-AA91-26258E1379D3.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018B-UL2018_MiniAODv2-v2/out_88.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/4D178EE9-2B49-4645-8C18-C27679980759.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018B-UL2018_MiniAODv2-v2/out_619.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/CEE18628-CCF5-F644-B146-61AEE0D9BF3E.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018B-UL2018_MiniAODv2-v2/out_397.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/AEBCD513-9CD6-5340-AFEF-07C17943D22A.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018B-UL2018_MiniAODv2-v2/out_396.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/2D40319C-51D2-7149-9789-D2178E3AA78D.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018B-UL2018_MiniAODv2-v2/out_395.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/7720DA1A-0A0B-B843-803E-D32DF9842F64.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018B-UL2018_MiniAODv2-v2/out_394.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/260000/8016643D-69DD-AC4B-BEB1-613446583135.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018B-UL2018_MiniAODv2-v2/out_82.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/260000/38C668EA-C687-6441-A42E-1A2D59D57D41.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018B-UL2018_MiniAODv2-v2/out_83.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/260000/362DB50F-D72E-C143-B7C3-F94D109BE675.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018B-UL2018_MiniAODv2-v2/out_80.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/260000/16E8B796-6294-3147-9CDA-3DCA55D0BD4E.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018B-UL2018_MiniAODv2-v2/out_81.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/260000/9BB0A1EF-61A3-8245-A041-ADCA9FF175CF.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018B-UL2018_MiniAODv2-v2/out_86.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/260000/2D2BE01E-8070-4B42-A924-D7C923A2C8E8.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018B-UL2018_MiniAODv2-v2/out_87.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/260000/56760187-7556-8947-B72B-2BA8BCDF2743.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018B-UL2018_MiniAODv2-v2/out_84.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/260000/BC73B7CC-E9F3-AF48-8846-E86CBB6939C3.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018B-UL2018_MiniAODv2-v2/out_85.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/110000/D0141784-EDA3-774F-9668-DE2B70D99323.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018B-UL2018_MiniAODv2-v2/out_797.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/110000/BE7F36C4-B0D8-E547-9489-3E3538F03778.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018B-UL2018_MiniAODv2-v2/out_796.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/110000/7CCCF7E8-A70A-0748-97C1-22BD0701E7EC.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018B-UL2018_MiniAODv2-v2/out_795.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/110000/DBF943CE-0738-204D-A74C-1ADC3BA3F6A1.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018B-UL2018_MiniAODv2-v2/out_794.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/110000/8C01D8F8-F73B-C249-89C8-A501FA94BC5E.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018B-UL2018_MiniAODv2-v2/out_793.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/110000/FC29DC71-3BD7-5842-95B4-A01FF3C789FF.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018B-UL2018_MiniAODv2-v2/out_792.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/110000/35327613-E2B2-3F4A-89D8-A68A0BF957DC.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018B-UL2018_MiniAODv2-v2/out_791.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/110000/35DB390D-F788-6B46-9140-D979247C5D7F.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018B-UL2018_MiniAODv2-v2/out_790.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/110000/596F5855-96B2-6946-9F7C-09F928E5F454.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018B-UL2018_MiniAODv2-v2/out_749.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/110000/E028F35A-0574-8A46-A5F0-D18FC7EAFF84.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018B-UL2018_MiniAODv2-v2/out_799.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/110000/6CB0E38F-3DD8-8C42-A847-B7B421EFB94E.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018B-UL2018_MiniAODv2-v2/out_798.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/260000/5ED30A86-7102-AC45-A0D7-A44ADC709E14.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018B-UL2018_MiniAODv2-v2/out_7.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/270000/9EC5879D-8344-EA4D-B374-3A3DB0EAC644.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018B-UL2018_MiniAODv2-v2/out_170.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/0867414C-498E-A94A-94CB-63025327098E.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018B-UL2018_MiniAODv2-v2/out_586.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/8300AA96-9455-3E4B-94FA-5E5A6F190807.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018B-UL2018_MiniAODv2-v2/out_587.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/8D87E9EE-177B-0140-A317-44BF47009B17.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018B-UL2018_MiniAODv2-v2/out_584.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/B311026A-BF32-DF49-BC46-EB9900F747AA.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018B-UL2018_MiniAODv2-v2/out_585.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/E110E9BC-9879-8849-B4B5-6AFF106B98A7.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018B-UL2018_MiniAODv2-v2/out_582.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/0BF6B61D-4633-A844-872A-2B4A0B4DABF6.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018B-UL2018_MiniAODv2-v2/out_583.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/14D0C166-59CF-1E4B-B134-B90B9D9655B5.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018B-UL2018_MiniAODv2-v2/out_580.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/6125BE66-AF66-CC41-BD67-5A15CBF09FED.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018B-UL2018_MiniAODv2-v2/out_581.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/74653DBC-930F-8445-9508-BDED506127C4.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018B-UL2018_MiniAODv2-v2/out_588.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/ED737496-A36D-C04C-8DE2-F31FE9FA5E0A.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018B-UL2018_MiniAODv2-v2/out_589.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/ED5EEEAE-9F05-2846-B2F0-10885ED404F4.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018B-UL2018_MiniAODv2-v2/out_245.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/601B7BC6-4D35-6048-8234-DB889A4FBC24.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018B-UL2018_MiniAODv2-v2/out_244.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/FB00AF75-EC2B-4A4E-A8C3-0080952FD3BC.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018B-UL2018_MiniAODv2-v2/out_247.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/9E056795-BC40-754C-B45B-3C528D4FD050.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018B-UL2018_MiniAODv2-v2/out_246.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/2EBDAB1C-8717-424F-98C9-4C0611F8C0F9.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018B-UL2018_MiniAODv2-v2/out_241.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/50E543C5-77EE-3746-8773-F83EB6E880C9.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018B-UL2018_MiniAODv2-v2/out_614.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/792E85B5-E646-AC4A-9D71-6F3C7D6133C6.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018B-UL2018_MiniAODv2-v2/out_243.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/649CB2E4-45E9-BC45-9E35-411FB56D064D.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018B-UL2018_MiniAODv2-v2/out_242.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/99C160B7-2736-834E-97DA-9E3496D85B85.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018B-UL2018_MiniAODv2-v2/out_615.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/3CBFB2DA-3F45-5043-B3DC-D5611A47623C.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018B-UL2018_MiniAODv2-v2/out_249.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/966DF944-DC20-D34E-88C3-6BD1452F46D2.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018B-UL2018_MiniAODv2-v2/out_248.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/7775C200-E22D-C64E-9732-FF04CFF24FC8.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018B-UL2018_MiniAODv2-v2/out_972.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/0AC46BEC-7AD3-6C44-829E-5EF99F269007.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018B-UL2018_MiniAODv2-v2/out_418.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/CC48EF75-CAE0-8F48-B856-C8B56FBCC5CE.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018B-UL2018_MiniAODv2-v2/out_224.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/2EDDCBA4-AC34-5640-B7F1-12CEC2D8C965.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018B-UL2018_MiniAODv2-v2/out_419.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/0FD766E3-1E77-1146-81BC-CD938F28AE37.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018B-UL2018_MiniAODv2-v2/out_519.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/12C03FA8-A320-2F46-B91E-1CFC2026CEAF.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018B-UL2018_MiniAODv2-v2/out_518.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/110000/2B71B3FE-5E8F-7E4A-B351-A6FC90DACA9E.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018B-UL2018_MiniAODv2-v2/out_1009.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/110000/B8B4CBEB-79F5-B448-B4C8-4D6F709BFD20.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018B-UL2018_MiniAODv2-v2/out_1008.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/110000/21EB5E7A-834E-B24C-A1FD-7D3FF3429BD6.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018B-UL2018_MiniAODv2-v2/out_1007.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/110000/3C4AEE4C-5D8A-0C41-8B32-B2F175E2E86F.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018B-UL2018_MiniAODv2-v2/out_1006.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/110000/4A250C25-1295-294B-98F9-F58A70161291.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018B-UL2018_MiniAODv2-v2/out_1005.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/110000/02BDD997-F0CF-B648-858E-E0D2016836F2.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018B-UL2018_MiniAODv2-v2/out_1004.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/B04ECC91-CE90-6F4C-A9C7-FAA195BE5578.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018B-UL2018_MiniAODv2-v2/out_515.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/110000/B8B9AE71-104C-CC4E-A5A5-1FF1F088C57C.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018B-UL2018_MiniAODv2-v2/out_1002.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/F165E78F-2646-2043-AF43-F554FEB0356E.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018B-UL2018_MiniAODv2-v2/out_517.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/110000/C4919F72-253A-4342-BB8E-1F8026AF6CC6.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018B-UL2018_MiniAODv2-v2/out_1000.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/1B4CD8B7-463F-FE45-B107-8B3A6AAE5CD6.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018B-UL2018_MiniAODv2-v2/out_623.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/D0B936A3-78B8-8C44-BCC6-A376C5FA38D9.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018B-UL2018_MiniAODv2-v2/out_622.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/16AF73CC-1E5D-D940-B3FF-124EB6764B18.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018B-UL2018_MiniAODv2-v2/out_621.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/ACA7BD97-8217-D647-B135-9F4AB2950605.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018B-UL2018_MiniAODv2-v2/out_514.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/1A5EF60F-352B-C043-B799-7F5571DFCD17.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018B-UL2018_MiniAODv2-v2/out_627.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/72F82A67-5DE5-974F-9DB0-06622026E5C4.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018B-UL2018_MiniAODv2-v2/out_626.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/DFD5A509-BDEB-654D-A745-C2C12C86C466.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018B-UL2018_MiniAODv2-v2/out_625.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/9D1A0490-A0DB-B346-A360-0969A609B139.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018B-UL2018_MiniAODv2-v2/out_624.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/C026750A-6A87-A343-85D3-766C879C1EC5.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018B-UL2018_MiniAODv2-v2/out_450.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/3ED771CE-AF27-FB40-BD46-5EA7DE7A57A6.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018B-UL2018_MiniAODv2-v2/out_451.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/F7958D3D-C5F4-764C-9B18-150BB15674B0.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018B-UL2018_MiniAODv2-v2/out_629.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/79AC42E5-41FA-1C41-8691-B02AA0B34AA5.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018B-UL2018_MiniAODv2-v2/out_628.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/6EAF757A-4820-A84C-AEBB-880FB557A1AB.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018B-UL2018_MiniAODv2-v2/out_454.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/6C1782DE-A639-3648-B5B6-806C3E1848C8.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018B-UL2018_MiniAODv2-v2/out_593.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/89ABB8BD-4DD7-1D46-A127-83310BAAE398.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018B-UL2018_MiniAODv2-v2/out_456.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/31692879-EBA5-BC4B-A426-07567B09B531.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018B-UL2018_MiniAODv2-v2/out_457.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/4B0CB9C0-0676-3949-8EB5-3B4BECCD7CCF.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018B-UL2018_MiniAODv2-v2/out_1084.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/270000/50A52F35-5C78-4843-9B50-CEF62AD0CB0F.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018B-UL2018_MiniAODv2-v2/out_179.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/270000/4EDA1CAD-31AF-0344-BEB2-B6A81416CB8B.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018B-UL2018_MiniAODv2-v2/out_178.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/270000/29E69823-6F4D-A644-9A73-3D076F695604.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018B-UL2018_MiniAODv2-v2/out_177.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/3D21F303-99E2-2E47-BD20-7B62D71C72BE.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018B-UL2018_MiniAODv2-v2/out_199.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/270000/157E9F94-DB0C-4B48-9BDF-8D0D92A5A79E.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018B-UL2018_MiniAODv2-v2/out_175.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/270000/62111FFB-8AD1-7D4B-8023-D08B35E6680C.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018B-UL2018_MiniAODv2-v2/out_174.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/270000/15C4E316-66E8-A44C-8AF0-7AFABCB6F8F8.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018B-UL2018_MiniAODv2-v2/out_173.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/270000/C73FC009-4297-544E-BB26-B7C4E667348B.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018B-UL2018_MiniAODv2-v2/out_172.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/270000/A25574DC-E7E8-3048-8418-347CF12D1D81.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018B-UL2018_MiniAODv2-v2/out_171.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/51B30CE4-7A39-DD4F-BFC7-12455D462084.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018B-UL2018_MiniAODv2-v2/out_594.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/6B705325-7CD6-264F-9A4E-05E6BC707427.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018B-UL2018_MiniAODv2-v2/out_977.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/270000/8E2D89EE-2864-B449-9816-F955BBA1DF0E.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018B-UL2018_MiniAODv2-v2/out_182.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/6FD515BF-C985-264B-881E-3AA639B8008F.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018B-UL2018_MiniAODv2-v2/out_975.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/4B698AAC-439B-914E-AC39-B339477C45D2.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018B-UL2018_MiniAODv2-v2/out_974.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/8D74FFC4-6055-CD48-973D-9F1EB495F5A7.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018B-UL2018_MiniAODv2-v2/out_973.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/9D15A17C-2E26-D341-94FF-DF8DA3EC6BC3.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018B-UL2018_MiniAODv2-v2/out_1081.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/CB52AC8E-12F5-304F-BA9F-0C750FFE255D.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018B-UL2018_MiniAODv2-v2/out_971.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/D1BB3594-3F32-774C-B999-AA0D3DE96D91.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018B-UL2018_MiniAODv2-v2/out_970.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/45472151-C26E-0E42-AC0A-B580C3C06FFB.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018B-UL2018_MiniAODv2-v2/out_1080.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/270000/E5912125-CC19-6C4A-9CAE-8953EB71524B.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018B-UL2018_MiniAODv2-v2/out_180.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/FDB327DA-2AE8-2941-B566-BA16059AD853.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018B-UL2018_MiniAODv2-v2/out_979.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/053097C2-5556-3C44-91E5-44A38B62A010.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018B-UL2018_MiniAODv2-v2/out_978.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/11457B7E-C699-964F-8BD7-B23868DFB74A.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018B-UL2018_MiniAODv2-v2/out_656.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/61389BFB-99FA-2542-A0F6-5DA999D010D8.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018B-UL2018_MiniAODv2-v2/out_599.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/4801B0F2-2160-BA43-AC81-2C3395D04A74.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018B-UL2018_MiniAODv2-v2/out_654.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/270000/E95BF3C6-4A18-354F-877E-794E2845DF39.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018B-UL2018_MiniAODv2-v2/out_181.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/83EF0EBC-A161-2948-B94A-A14B27CD52F0.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018B-UL2018_MiniAODv2-v2/out_186.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/201432CD-0E57-604B-AA2E-9A67A77E3B4D.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018B-UL2018_MiniAODv2-v2/out_187.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/902EA656-0371-4E4D-A490-FD0A2FCC6722.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018B-UL2018_MiniAODv2-v2/out_184.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/BAFAC8C7-9E91-0942-99EF-9518B8762853.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018B-UL2018_MiniAODv2-v2/out_185.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/06058C67-14D9-F34B-84D0-21A4A8057D5C.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018B-UL2018_MiniAODv2-v2/out_652.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/5224B6A1-9444-C448-BB96-3AE24D6A5007.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018B-UL2018_MiniAODv2-v2/out_188.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/E1913841-900E-004D-BAFA-DE5BB6385D26.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018B-UL2018_MiniAODv2-v2/out_189.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/6BC19497-20F7-604B-BFA2-CFD2C0B87A33.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018B-UL2018_MiniAODv2-v2/out_658.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/744D9377-F7D6-114E-A85C-60FD3647309A.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018B-UL2018_MiniAODv2-v2/out_516.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/8E7E8742-649F-514F-816E-B0E6DD683B81.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018B-UL2018_MiniAODv2-v2/out_650.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/AA6A5A1D-07B6-D943-8BD4-C0BBB15C9C93.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018B-UL2018_MiniAODv2-v2/out_651.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/FCF3F138-AB31-9C4F-A5BA-1356C61A274C.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018B-UL2018_MiniAODv2-v2/out_1089.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/2A14E917-E37B-1444-91D7-CD2E04E12614.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018B-UL2018_MiniAODv2-v2/out_1088.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/260000/9DB70BE1-CEC7-1743-9C6F-C26635733A2A.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018B-UL2018_MiniAODv2-v2/out_11.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/260000/F99F37E5-E576-D34C-B7AD-5DF165DC9D31.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018B-UL2018_MiniAODv2-v2/out_10.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/260000/357F5BCA-7BFE-F54F-820E-43D18E5DD6B9.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018B-UL2018_MiniAODv2-v2/out_13.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/260000/B9E6C7B7-4D24-8446-B562-44B649AF80B0.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018B-UL2018_MiniAODv2-v2/out_12.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/260000/48BACDC6-28F0-854A-8B9F-240245D02E5B.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018B-UL2018_MiniAODv2-v2/out_15.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/260000/FEC7D99E-C234-7346-BF7C-4159B8A17F4D.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018B-UL2018_MiniAODv2-v2/out_14.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/260000/1A123346-5C89-CF4B-9BA0-7BD845C80C99.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018B-UL2018_MiniAODv2-v2/out_17.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/260000/119D1B71-0A86-E444-BACE-E9FC2F530C3E.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018B-UL2018_MiniAODv2-v2/out_16.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/260000/89449701-3696-D44F-98D7-DE16901F0F7C.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018B-UL2018_MiniAODv2-v2/out_19.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/260000/C31A0081-8C0E-504B-AAA4-BA3DCB20ED25.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018B-UL2018_MiniAODv2-v2/out_18.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/2EFCD359-301C-E44A-95B4-6C68B94B9FEE.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018B-UL2018_MiniAODv2-v2/out_863.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/14C6D0A4-5F26-D44C-AF36-A6EA3B510F1A.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018B-UL2018_MiniAODv2-v2/out_862.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/24273755-1D1F-F143-A8BF-1938577DE856.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018B-UL2018_MiniAODv2-v2/out_865.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/81C6E1E8-1667-E247-89C2-30422F1BE6A7.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018B-UL2018_MiniAODv2-v2/out_864.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/B8927EF7-4B3D-6F47-9EB6-5704BC4046C4.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018B-UL2018_MiniAODv2-v2/out_867.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/85343008-E62B-8044-BA9D-9A4D155899A4.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018B-UL2018_MiniAODv2-v2/out_866.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70002/A54F80F5-24FC-694E-87F9-2507A4B89A49.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018B-UL2018_MiniAODv2-v2/out_354.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/51210D34-AB14-ED4D-B96D-7D1D119CD3C4.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018B-UL2018_MiniAODv2-v2/out_659.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/260000/15DF5226-3FCC-FC44-93DA-31666B6A8BC7.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018B-UL2018_MiniAODv2-v2/out_44.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/110000/71209C33-63F1-A941-9DBF-A667EC2A7AD5.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018B-UL2018_MiniAODv2-v2/out_883.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/110000/F671022F-D13D-A943-867B-DD8915C67738.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018B-UL2018_MiniAODv2-v2/out_882.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/110000/81D0A47F-E872-9843-8F92-6B245507794D.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018B-UL2018_MiniAODv2-v2/out_881.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/110000/AA244B04-3D15-C349-8049-375953DD4CBD.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018B-UL2018_MiniAODv2-v2/out_880.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/110000/4C38CD35-ADA8-664A-9520-9FFB1AC0EECE.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018B-UL2018_MiniAODv2-v2/out_887.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/110000/586B31B3-DD9A-C54F-83D7-2642B2E368DD.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018B-UL2018_MiniAODv2-v2/out_886.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/110000/E65BB6E0-5DFB-B442-9E85-2BEBC719ED03.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018B-UL2018_MiniAODv2-v2/out_885.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/110000/A5985176-A99C-9F43-B4F8-8A40F612455E.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018B-UL2018_MiniAODv2-v2/out_884.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/110000/E9CF9A7C-35A7-4344-901D-DCBCE56E26FE.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018B-UL2018_MiniAODv2-v2/out_889.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/110000/C3E2E62F-7302-A144-BCBD-2BA6D7E63BFA.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018B-UL2018_MiniAODv2-v2/out_888.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/260000/F1267683-7B71-AE49-A834-A33B4C43C2DE.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018B-UL2018_MiniAODv2-v2/out_116.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/260000/4A005C73-C443-F043-9FF1-0213AB73E20B.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018B-UL2018_MiniAODv2-v2/out_45.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/4D9D6F19-9B43-684F-B9F6-C921874D0152.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018B-UL2018_MiniAODv2-v2/out_657.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/21AB9A34-5088-D942-9680-E82DA29A0137.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018B-UL2018_MiniAODv2-v2/out_355.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/A8C23BBE-6F01-8441-B51C-D2E3F3EB3337.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018B-UL2018_MiniAODv2-v2/out_322.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/88675FC2-44E8-1B42-A05E-276FBC969412.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018B-UL2018_MiniAODv2-v2/out_323.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/16CC1154-425F-484A-B421-EA9236E7ED29.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018B-UL2018_MiniAODv2-v2/out_320.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/8B2C0B0E-C952-914D-9F88-D88621373D09.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018B-UL2018_MiniAODv2-v2/out_321.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/10DEA711-2FB8-7840-A038-087FF0FFDAF5.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018B-UL2018_MiniAODv2-v2/out_326.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/6D59C6CB-9805-104D-8ADE-6DFA124A29EF.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018B-UL2018_MiniAODv2-v2/out_327.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/B7B26D7A-620F-4E4F-9DFF-C8895DE61239.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018B-UL2018_MiniAODv2-v2/out_324.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/FE8DB9C9-F90F-8842-B3B7-1C020DF473E4.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018B-UL2018_MiniAODv2-v2/out_325.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/8B7FACBE-A327-734E-A543-329D5D80EBA3.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018B-UL2018_MiniAODv2-v2/out_328.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/A4EBDF76-D22B-3249-87AA-9C2FED381872.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018B-UL2018_MiniAODv2-v2/out_329.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/7ACFA4AB-AC2D-7041-95EE-6A22A323AA20.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018B-UL2018_MiniAODv2-v2/out_562.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/8010AADA-7395-124B-BD03-573FE5F5C2AD.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018B-UL2018_MiniAODv2-v2/out_201.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/82AFB835-C7F8-5441-984B-A74A2AB16BA6.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018B-UL2018_MiniAODv2-v2/out_200.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/E78E6EEB-E30B-084A-9BD4-F8B9A587D7FC.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018B-UL2018_MiniAODv2-v2/out_203.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/61818D54-4D85-5B43-95CE-C0CDF47C9DB1.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018B-UL2018_MiniAODv2-v2/out_202.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/397A43DA-11B2-C845-8A5D-CE5359A1D175.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018B-UL2018_MiniAODv2-v2/out_205.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/21F38C5C-565D-D347-96D7-8E1E8478A5FD.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018B-UL2018_MiniAODv2-v2/out_204.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/88C12AC7-D527-0145-95D6-684878D4B388.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018B-UL2018_MiniAODv2-v2/out_207.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/E1E83BD2-0EF2-764C-AA57-5593BEEB20BA.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018B-UL2018_MiniAODv2-v2/out_206.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/0387FD6A-C5BA-C647-8C93-A44C5B342BDE.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018B-UL2018_MiniAODv2-v2/out_209.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/A21E3112-2DEA-4D42-9B52-09C415DF4DE2.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018B-UL2018_MiniAODv2-v2/out_208.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/110000/CADC386B-3D51-364C-8636-6B35D78EF053.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018B-UL2018_MiniAODv2-v2/out_779.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/110000/D629CDB7-D5E1-4A4F-99DC-E0A58EF71448.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018B-UL2018_MiniAODv2-v2/out_778.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/260000/61973643-0CA2-BD4D-920D-54E0EB5339E8.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018B-UL2018_MiniAODv2-v2/out_77.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/260000/A30536AF-C514-D449-BCA8-F1792E0BDE2B.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018B-UL2018_MiniAODv2-v2/out_76.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/260000/C6180AE2-344F-0E4F-B569-704B1041C9A9.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018B-UL2018_MiniAODv2-v2/out_75.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/260000/5D79C000-49E4-1944-B747-56DFC7F7DEEF.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018B-UL2018_MiniAODv2-v2/out_74.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/260000/3C7DA8A1-FDFC-9249-8F1C-5BE1776D73E6.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018B-UL2018_MiniAODv2-v2/out_73.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/260000/497F1937-EF27-DF43-9A5B-0B2D1F2D31E1.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018B-UL2018_MiniAODv2-v2/out_72.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/260000/FD6DB387-1D58-F94A-9574-B810765E85EC.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018B-UL2018_MiniAODv2-v2/out_71.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/260000/B1FC5E63-339D-3E42-A303-7BBCAFBEF304.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018B-UL2018_MiniAODv2-v2/out_70.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/1F4DDEE0-89CD-B249-9C3C-A0ACE8EA3D58.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018B-UL2018_MiniAODv2-v2/out_655.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/83417DD3-BFC5-9C4D-8574-9CFD374FAE1B.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018B-UL2018_MiniAODv2-v2/out_567.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/260000/39F80533-3ED0-E94B-B887-D4AFC43C45B2.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018B-UL2018_MiniAODv2-v2/out_79.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/260000/F4304A15-D698-D047-9CB0-2A41DCCE6332.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018B-UL2018_MiniAODv2-v2/out_78.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/110000/027C0A31-BD7E-F04A-99C0-067C4F606BD1.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018B-UL2018_MiniAODv2-v2/out_1043.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/110000/20A201A1-3C6C-8847-8ECA-3B83A627912E.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018B-UL2018_MiniAODv2-v2/out_1042.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/110000/59F7D27B-1E9C-454E-9442-53070ADC0847.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018B-UL2018_MiniAODv2-v2/out_1041.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/110000/92F0136E-4818-D349-AB11-204A41D575C8.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018B-UL2018_MiniAODv2-v2/out_1040.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/110000/884CEBA6-0634-C241-ABE3-2D3639EDC8B7.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018B-UL2018_MiniAODv2-v2/out_1047.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/110000/51172827-671F-F448-AC90-47F67E0F0B98.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018B-UL2018_MiniAODv2-v2/out_1046.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/110000/F3C741A2-A2DC-7240-A86B-1A1B61C0C261.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018B-UL2018_MiniAODv2-v2/out_1045.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/110000/37AEF022-73CD-0747-AC8E-0824BEBCD494.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018B-UL2018_MiniAODv2-v2/out_1044.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/110000/285CC408-058E-B448-A08E-1B45EC949AE8.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018B-UL2018_MiniAODv2-v2/out_1049.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/110000/C6381A45-6E90-134E-B6E7-5F97DDE48F6B.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018B-UL2018_MiniAODv2-v2/out_1048.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/5889724F-4EFC-DA45-9908-690B0E357C0A.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018B-UL2018_MiniAODv2-v2/out_359.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/D54D10ED-B79A-2F4F-9195-28BED4C2A2A0.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018B-UL2018_MiniAODv2-v2/out_358.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/1A4F9AA6-E4B1-CB4C-A987-013C0ACCC060.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018B-UL2018_MiniAODv2-v2/out_868.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/8EA9FB25-0E60-684B-9EEE-39DDC6B70F0F.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018B-UL2018_MiniAODv2-v2/out_669.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/76DE7A9E-388F-8440-BF1E-7453125EC3D8.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018B-UL2018_MiniAODv2-v2/out_668.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/281A19E4-D8E4-FE4E-BF92-80BEEA95E39A.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018B-UL2018_MiniAODv2-v2/out_667.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/B0C5AB27-7F27-EC44-9777-EDBDDBC04D6C.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018B-UL2018_MiniAODv2-v2/out_666.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/F792815A-2D09-9246-82AB-0A0316EE0051.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018B-UL2018_MiniAODv2-v2/out_665.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/D138D930-9B79-4246-ADB6-82B340E556FD.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018B-UL2018_MiniAODv2-v2/out_664.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/1CB0E797-2B1D-164D-AA83-78F67CEF1E8E.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018B-UL2018_MiniAODv2-v2/out_663.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/C6E42883-402A-0E46-A549-BE54B577A13E.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018B-UL2018_MiniAODv2-v2/out_662.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/7655D3DB-4213-9F44-93A1-3AA0123CD121.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018B-UL2018_MiniAODv2-v2/out_661.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/F3CC2B42-8A5F-6F4B-B9E3-F5B4A41C2758.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018B-UL2018_MiniAODv2-v2/out_660.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/66B49188-687C-BA4E-9059-EA56C0BBC2BB.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018B-UL2018_MiniAODv2-v2/out_215.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/4FDC0C36-BE79-DF4C-8C90-B54B340368B3.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018B-UL2018_MiniAODv2-v2/out_653.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/C20544F8-AF3C-DC41-A05F-29799417C330.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018B-UL2018_MiniAODv2-v2/out_692.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/2CF293CC-4791-E344-819F-9C3ECCA17074.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018B-UL2018_MiniAODv2-v2/out_693.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/D2E9490E-6DC5-D649-9793-4F1FABE5AA89.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018B-UL2018_MiniAODv2-v2/out_690.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/D8FC1913-76A9-834A-AD33-64F6F96BC92C.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018B-UL2018_MiniAODv2-v2/out_691.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/9FB29AE1-D177-644E-A692-364A6608FAEE.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018B-UL2018_MiniAODv2-v2/out_696.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/8519EDAC-0A82-E640-9AC3-B6FB98746261.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018B-UL2018_MiniAODv2-v2/out_697.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/D7DD8913-3597-7248-BF5D-218167B5E88D.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018B-UL2018_MiniAODv2-v2/out_694.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/9ED11561-37D0-A847-8FA1-E35849C7E1B0.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018B-UL2018_MiniAODv2-v2/out_695.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/01C2CBA6-2341-2A47-AA01-BEFF319CB471.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018B-UL2018_MiniAODv2-v2/out_698.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/320DCDC5-5ADE-6A44-B144-73151C6F7172.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018B-UL2018_MiniAODv2-v2/out_699.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/0F399C52-5BC0-704D-89EA-FD38BE3B3506.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018B-UL2018_MiniAODv2-v2/out_222.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/9FAE3B8A-164F-AC4D-B0BD-EB0CCB1F27E6.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018B-UL2018_MiniAODv2-v2/out_542.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/27DEA455-F913-B34F-9F9D-81B77481857C.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018B-UL2018_MiniAODv2-v2/out_543.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/C89A21D2-B144-CA4A-AEE6-2CF3764F41DB.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018B-UL2018_MiniAODv2-v2/out_540.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/74214170-4DC3-3447-AD1D-57B0369BF668.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018B-UL2018_MiniAODv2-v2/out_541.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/A83BCF9C-230E-0D41-BE26-0B4DC7CC2633.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018B-UL2018_MiniAODv2-v2/out_546.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/38431468-FA8C-AE47-A57F-5C433A03B20F.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018B-UL2018_MiniAODv2-v2/out_547.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/448B1052-84E3-1847-862E-7B6E5DE866EB.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018B-UL2018_MiniAODv2-v2/out_544.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/C25744F0-DA83-9B4F-880A-FB536154E56D.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018B-UL2018_MiniAODv2-v2/out_545.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/110000/555EA6D2-3B7C-1B45-BBEA-E82E04A3755F.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018B-UL2018_MiniAODv2-v2/out_1018.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/E172C59F-B23A-4A48-8E6B-25C832B569FD.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018B-UL2018_MiniAODv2-v2/out_548.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/282FCDFC-A21B-9046-98BE-11351B97F0A4.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018B-UL2018_MiniAODv2-v2/out_549.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/110000/52969FD1-47DA-C949-9824-CEEB25267D01.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018B-UL2018_MiniAODv2-v2/out_1019.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/110000/C8887C30-7D5E-CA46-8D2A-E92510DE0892.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018B-UL2018_MiniAODv2-v2/out_995.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/110000/A0E8BBDC-0B23-7C44-A3BE-C5C115809304.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018B-UL2018_MiniAODv2-v2/out_994.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/110000/1560B4D4-F674-B84B-B833-01019FD2AA00.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018B-UL2018_MiniAODv2-v2/out_997.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/110000/615814AE-64D2-8C4A-93D3-283BA8F259A2.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018B-UL2018_MiniAODv2-v2/out_996.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/110000/83F0AE9C-8A1A-DF41-ADD3-45D39B7E4530.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018B-UL2018_MiniAODv2-v2/out_991.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/110000/61D835A4-0B62-BD43-8949-ABFFA9AC59AA.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018B-UL2018_MiniAODv2-v2/out_990.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/110000/AE0102DF-D1D6-B642-90C5-3C1C383E44B2.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018B-UL2018_MiniAODv2-v2/out_993.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/110000/37B932A5-5220-7741-ACA9-9179915804F8.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018B-UL2018_MiniAODv2-v2/out_992.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/110000/73319725-7D0F-B743-986A-862C1CF1F956.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018B-UL2018_MiniAODv2-v2/out_999.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/110000/589ADB9E-0059-C14C-AD03-4B9CABB1FCF3.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018B-UL2018_MiniAODv2-v2/out_998.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/88748742-6193-6546-998F-6FA868D2FC0E.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018B-UL2018_MiniAODv2-v2/out_120.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/C409F8C8-744A-A641-AFE6-E97CBDD0E856.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018B-UL2018_MiniAODv2-v2/out_121.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/7F654A01-CE57-5047-B7A4-92CF5B1C6EDA.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018B-UL2018_MiniAODv2-v2/out_122.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/CE83E288-9F27-7B48-B6F7-5D4E467C913E.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018B-UL2018_MiniAODv2-v2/out_123.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/E022CD5D-9D37-7649-89B9-967F0200F6EB.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018B-UL2018_MiniAODv2-v2/out_124.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/61FD2835-447C-C140-BD57-5505B23ACA85.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018B-UL2018_MiniAODv2-v2/out_125.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/9AD4C657-CFEE-2640-9E2D-ACAA5827DE60.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018B-UL2018_MiniAODv2-v2/out_126.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/9D24F5B0-0BCB-7B4D-9309-682B7D507FC7.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018B-UL2018_MiniAODv2-v2/out_127.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/66CD97E9-4339-5342-84C3-71DFDA7D1FA3.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018B-UL2018_MiniAODv2-v2/out_128.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/260000/8064ACEA-5106-8145-9A3E-4597C39D0F24.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018B-UL2018_MiniAODv2-v2/out_2.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/4047FBFC-8332-BA49-8F99-7FB8DAF8337A.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018B-UL2018_MiniAODv2-v2/out_219.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/270000/26D8494A-8BA5-C54C-98F9-0A486373EF19.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018B-UL2018_MiniAODv2-v2/out_176.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/458A795D-6BC6-244C-B7D6-AAC3B39AF832.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018B-UL2018_MiniAODv2-v2/out_214.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/110000/7485BBBD-0C4C-6543-ADF7-A50AF4B0ED1C.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018B-UL2018_MiniAODv2-v2/out_777.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/110000/CB6B9683-7A88-964F-A714-977A9D37F806.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018B-UL2018_MiniAODv2-v2/out_1010.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/249E7BB0-0F7E-6447-AC7B-14C1D7149263.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018B-UL2018_MiniAODv2-v2/out_563.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/110000/64D65686-191D-8B48-ADB9-A350FA62EB02.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018B-UL2018_MiniAODv2-v2/out_1022.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/110000/63856A95-FAF3-8B4F-B227-37DEC49F1E87.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018B-UL2018_MiniAODv2-v2/out_1011.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/110000/049FCC4B-DE28-5B4D-8ECB-00E15EF946EA.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018B-UL2018_MiniAODv2-v2/out_928.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/110000/7A70F508-F9CD-AC4C-8C0C-475A8D89EA45.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018B-UL2018_MiniAODv2-v2/out_929.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/CCB55D97-36E5-3C42-953A-6F9CCFD896D6.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018B-UL2018_MiniAODv2-v2/out_416.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/860ABB63-6463-0E44-8AA6-2F995624674D.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018B-UL2018_MiniAODv2-v2/out_417.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/6CFEC282-8E61-6A45-9263-4D8AED9EE33C.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018B-UL2018_MiniAODv2-v2/out_410.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/B16F2F48-1097-1649-A0EC-D21B1A75CE57.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018B-UL2018_MiniAODv2-v2/out_411.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/565780C2-858C-4C4F-9FB2-A271E76458EB.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018B-UL2018_MiniAODv2-v2/out_412.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/FBFE03B1-6D01-384C-B6A6-880C792BCB79.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018B-UL2018_MiniAODv2-v2/out_413.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/110000/5696C1E3-A722-BE46-AE55-6DDAD04C9024.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018B-UL2018_MiniAODv2-v2/out_920.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/D64E2C25-625D-2344-89BB-CDA76BFE03B6.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018B-UL2018_MiniAODv2-v2/out_498.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/110000/FC2F92C6-7244-154F-985C-3687713F7823.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018B-UL2018_MiniAODv2-v2/out_922.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/110000/FDC8092C-CD4C-D844-AFB7-9CD81D29B858.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018B-UL2018_MiniAODv2-v2/out_923.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/110000/BD1EF09E-07FB-924F-A97C-584C2C690B9E.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018B-UL2018_MiniAODv2-v2/out_924.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/110000/AA90087D-D7CB-044C-93C8-593C84B52B70.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018B-UL2018_MiniAODv2-v2/out_925.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/110000/2E86EEA7-DC76-2343-9B6F-66F36E555AC1.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018B-UL2018_MiniAODv2-v2/out_926.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/110000/A135CBE2-C697-E542-A626-D3C2787376E5.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018B-UL2018_MiniAODv2-v2/out_927.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/1A3E21AE-AEF5-2C41-B609-E7451E3430CE.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018B-UL2018_MiniAODv2-v2/out_319.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/58EC0861-26FB-C242-B928-C9BE725C1850.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018B-UL2018_MiniAODv2-v2/out_318.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/6E53CDFE-3E64-6C4D-AA10-284509624A5A.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018B-UL2018_MiniAODv2-v2/out_313.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/F9EEC6E5-C30D-8E4A-B099-60FD74589D6F.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018B-UL2018_MiniAODv2-v2/out_312.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/64FC6D1F-1818-AB42-AA00-1E7320F3F1CC.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018B-UL2018_MiniAODv2-v2/out_311.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/A141DAA9-1140-7A48-950D-05EF2520A7EF.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018B-UL2018_MiniAODv2-v2/out_310.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/50E40ED2-E314-7648-892D-7BEA2BD9D632.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018B-UL2018_MiniAODv2-v2/out_317.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/62B00A21-A497-4544-8B57-DA9C29328419.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018B-UL2018_MiniAODv2-v2/out_316.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/D4FA02CA-A04D-234A-A7BB-BD91D2C3999F.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018B-UL2018_MiniAODv2-v2/out_315.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/83743AD9-2AF7-CF45-A5F9-EE567E8B6EFB.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018B-UL2018_MiniAODv2-v2/out_314.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/110000/3F734C92-3056-B943-B47C-6E6C614E712C.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018B-UL2018_MiniAODv2-v2/out_921.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/D816C7DE-BE5E-8D4C-959E-3FB0B0754B38.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018B-UL2018_MiniAODv2-v2/out_496.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/D0BAD346-EDC1-B34E-A90E-A225FF4FE699.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018B-UL2018_MiniAODv2-v2/out_670.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/377A9C19-9189-8C47-B457-04BA6B728460.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018B-UL2018_MiniAODv2-v2/out_832.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/DB78B2DC-EB92-4D4A-A5B6-D06E4E3785E1.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018B-UL2018_MiniAODv2-v2/out_833.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/0BBF40F1-F63A-5742-8412-EBBAD72361AB.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018B-UL2018_MiniAODv2-v2/out_830.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/BC3D31F9-47C2-B445-B95D-BEB7C18DAFE2.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018B-UL2018_MiniAODv2-v2/out_831.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/E7A8F65A-F97A-2545-B610-DCFBB34A525A.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018B-UL2018_MiniAODv2-v2/out_836.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/B6CE147A-0A40-6F44-A22A-AF55ED69201C.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018B-UL2018_MiniAODv2-v2/out_837.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/07ED9272-B1DB-464D-B975-424D269FBB7E.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018B-UL2018_MiniAODv2-v2/out_834.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/F748D3C9-CFA6-C046-BC8F-D1BB406FEE49.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018B-UL2018_MiniAODv2-v2/out_835.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/458ED136-EAAD-8A48-9445-41EF638D1B41.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018B-UL2018_MiniAODv2-v2/out_838.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/57D9E12A-6F33-074A-9029-96C74A4BB01B.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018B-UL2018_MiniAODv2-v2/out_839.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/260000/62CBD66E-19ED-5A49-9588-F1CF4CB52433.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018B-UL2018_MiniAODv2-v2/out_3.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/16D94F31-442D-E648-845F-57DB12D51A85.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018B-UL2018_MiniAODv2-v2/out_725.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/37AF67EF-5F71-4A4C-A1B1-924922199581.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018B-UL2018_MiniAODv2-v2/out_368.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/D8AFA6BB-7077-5540-BA02-9EAF75870836.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018B-UL2018_MiniAODv2-v2/out_369.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/5C3D9024-B6EE-B143-8918-168610476FF9.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018B-UL2018_MiniAODv2-v2/out_366.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/1EA3C6EB-FFE2-6D4F-AB50-2F01B4E946D3.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018B-UL2018_MiniAODv2-v2/out_367.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/3F587069-7D84-3440-91B1-E2C88243E74C.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018B-UL2018_MiniAODv2-v2/out_364.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/76D0DE1E-004D-4A4A-B0BD-D3AA9E1D1561.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018B-UL2018_MiniAODv2-v2/out_365.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/6AF3E88D-CFCD-8D4A-8CE5-5147DA51C5EB.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018B-UL2018_MiniAODv2-v2/out_362.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/D5D8B6E8-43E6-E848-AE61-21BAAB6EABF1.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018B-UL2018_MiniAODv2-v2/out_363.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/D2865EE7-2827-4147-8EFE-52F832E46A10.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018B-UL2018_MiniAODv2-v2/out_360.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/8CAD1C45-6269-6C45-B456-D3FEE6EEB703.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018B-UL2018_MiniAODv2-v2/out_361.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/6B851661-5C65-EC40-B11A-C8A8411C2DAF.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018B-UL2018_MiniAODv2-v2/out_218.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/F775B26A-420E-5B44-A372-278083EAF0EC.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018B-UL2018_MiniAODv2-v2/out_380.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/66D2E76C-FD9C-D04E-89BF-DBD1C60CEC72.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018B-UL2018_MiniAODv2-v2/out_861.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/27F9E115-D447-F34F-9784-1230F13ACDD9.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018B-UL2018_MiniAODv2-v2/out_382.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/E7CA16B3-40F4-CF4C-B377-96AB1CAA9E86.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018B-UL2018_MiniAODv2-v2/out_383.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/C9F36C15-9E48-6847-8663-8684303B5851.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018B-UL2018_MiniAODv2-v2/out_384.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/8CB19F76-F50A-9348-967C-3DFA3710DAB9.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018B-UL2018_MiniAODv2-v2/out_385.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/954DC8F9-7692-2643-8CAB-44B209D499AE.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018B-UL2018_MiniAODv2-v2/out_386.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/152124A0-3487-E54D-8F7D-B4E7689D4DA0.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018B-UL2018_MiniAODv2-v2/out_387.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/A7997A5F-13D9-684F-B3EE-7847B67643F7.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018B-UL2018_MiniAODv2-v2/out_388.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/FCC0830E-458A-264C-AD73-097A2F6BFE29.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018B-UL2018_MiniAODv2-v2/out_389.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/110000/55BC9B4C-B37C-FF40-B361-20FC213BD0E1.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018B-UL2018_MiniAODv2-v2/out_784.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/110000/BFA2FD6F-2AE9-1741-BC93-1161223EF116.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018B-UL2018_MiniAODv2-v2/out_785.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/110000/F1A28656-3E06-F044-8E3C-39B5B7AD7E5C.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018B-UL2018_MiniAODv2-v2/out_786.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/110000/A6CD8CB4-8836-C04C-A906-1A7CD4AF9A47.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018B-UL2018_MiniAODv2-v2/out_787.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/110000/EC6CC4BC-BF9F-EE43-BACB-7EF6DC0DEB9A.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018B-UL2018_MiniAODv2-v2/out_780.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/110000/71CF349A-95BA-1049-8779-6ABC87172690.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018B-UL2018_MiniAODv2-v2/out_781.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/110000/69F0F14B-1D88-354B-9EF6-865CD1D6D8DE.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018B-UL2018_MiniAODv2-v2/out_782.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/BC157E2F-7894-B14B-845E-1B6F489EEF7F.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018B-UL2018_MiniAODv2-v2/out_381.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/110000/82EAA596-E42D-0840-B378-2B39F578065E.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018B-UL2018_MiniAODv2-v2/out_788.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/110000/2A7F9B76-2881-A344-A7E6-B1B0ED5634EC.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018B-UL2018_MiniAODv2-v2/out_789.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/B9240ED4-3240-F94D-8521-ABDC7F60A0E8.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018B-UL2018_MiniAODv2-v2/out_860.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/FF626DF8-A706-1142-B934-53E1594E6B77.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018B-UL2018_MiniAODv2-v2/out_151.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/F4199A53-EAB4-A442-9337-E810F92E6322.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018B-UL2018_MiniAODv2-v2/out_579.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/128A5B72-BA3C-5140-8503-FC1B36DBE6EA.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018B-UL2018_MiniAODv2-v2/out_578.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/3DC53A6A-2BFB-EA4B-9CB2-0E632369C86E.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018B-UL2018_MiniAODv2-v2/out_150.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/B21200BA-8CA7-224C-8C38-30394F8E8F61.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018B-UL2018_MiniAODv2-v2/out_573.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/CEBDC5D8-E57F-AD41-BC26-FF5172495ABB.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018B-UL2018_MiniAODv2-v2/out_572.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/5365FEC0-B4AD-A445-81D4-038ADA344DEF.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018B-UL2018_MiniAODv2-v2/out_571.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/95253980-A8D8-AF49-8CBA-CF52E88F94DB.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018B-UL2018_MiniAODv2-v2/out_570.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/E8D0B23E-DB8C-F049-BF8E-FB54C962757A.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018B-UL2018_MiniAODv2-v2/out_577.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/843D8DB2-2701-444C-910B-BB76F9089D9A.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018B-UL2018_MiniAODv2-v2/out_576.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/F4A02460-B2B5-1845-8AFB-1A789E7B86A2.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018B-UL2018_MiniAODv2-v2/out_575.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/05F657B4-399B-BC46-ABDD-8135CFB6538E.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018B-UL2018_MiniAODv2-v2/out_574.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/260000/2799C58B-391E-994A-B423-76088CCC5052.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018B-UL2018_MiniAODv2-v2/out_60.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/260000/9725AD67-E606-5F4E-B040-DA41F6F9EAB5.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018B-UL2018_MiniAODv2-v2/out_61.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/260000/88C5E818-0E69-7E4B-BC00-53EB6E3156BE.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018B-UL2018_MiniAODv2-v2/out_62.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/521173F0-33D6-AD4E-B967-D6DFF5E54C1E.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018B-UL2018_MiniAODv2-v2/out_259.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/260000/8F45D494-0CD7-C54C-B2F3-3F6A9E1B5C26.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018B-UL2018_MiniAODv2-v2/out_64.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/260000/9FA5A7EA-8046-564A-BBBD-F9A078EB9FCF.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018B-UL2018_MiniAODv2-v2/out_65.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/260000/82ACEB4E-A65D-854B-B907-EABCBF82156A.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018B-UL2018_MiniAODv2-v2/out_66.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/260000/4632DE63-B823-C54F-93AE-C395744B1E18.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018B-UL2018_MiniAODv2-v2/out_67.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/260000/BDB8C0B5-569D-0845-A07F-EA479B8F9999.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018B-UL2018_MiniAODv2-v2/out_68.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/5085C66D-1896-6542-8685-4A184834F671.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018B-UL2018_MiniAODv2-v2/out_253.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70002/54A9DB7D-122D-F046-B99D-199B1480C033.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018B-UL2018_MiniAODv2-v2/out_250.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/DA9F9399-B810-8A40-B3D8-37809E436811.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018B-UL2018_MiniAODv2-v2/out_251.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/B6B4E05B-6413-764E-B678-FDDC768E0A28.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018B-UL2018_MiniAODv2-v2/out_256.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/567FDD6C-C2E3-AD49-B936-1A9048BF8081.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018B-UL2018_MiniAODv2-v2/out_257.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/2DB2967D-DA17-1F4C-9EA0-DC12224F9CE5.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018B-UL2018_MiniAODv2-v2/out_254.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/11FADE68-FCA1-9146-B3DD-4293A85C8DD3.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018B-UL2018_MiniAODv2-v2/out_255.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/C84DA056-E3C2-1D4B-8906-CE1824755122.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018B-UL2018_MiniAODv2-v2/out_499.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/998B5824-B596-E34D-9C4D-CF951E344E51.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018B-UL2018_MiniAODv2-v2/out_157.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/A1A4D00F-15A9-7A45-AF30-6AA54C4ADBD1.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018B-UL2018_MiniAODv2-v2/out_156.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/110000/CAE61857-77AC-CD4B-937C-FF0429E13456.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018B-UL2018_MiniAODv2-v2/out_939.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/110000/0E4DACB9-C881-EA4A-A66B-C5DEB35B6774.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018B-UL2018_MiniAODv2-v2/out_731.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/110000/358CE3A2-B592-4E43-8AE3-8FC08272293B.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018B-UL2018_MiniAODv2-v2/out_730.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/110000/7300E713-5F40-BA48-B722-37F8881AE70E.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018B-UL2018_MiniAODv2-v2/out_733.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/110000/89A328F6-AB6F-8649-B395-86C36B4CB879.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018B-UL2018_MiniAODv2-v2/out_938.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/110000/D4650568-E8A9-6846-BC22-6158912D438B.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018B-UL2018_MiniAODv2-v2/out_735.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/110000/9E89AF0E-3E1E-6647-AC1E-83B1A6C16218.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018B-UL2018_MiniAODv2-v2/out_734.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/6BECDFDD-9AD0-5E4D-B652-247E99B57754.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018B-UL2018_MiniAODv2-v2/out_508.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/110000/88496D78-2467-774E-805C-E8C7EAEFA616.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018B-UL2018_MiniAODv2-v2/out_736.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/110000/145B7BFF-8EEE-5448-B614-EC836C415857.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018B-UL2018_MiniAODv2-v2/out_1032.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/110000/ACCC2277-928E-2E43-A64B-1A00FAE27F85.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018B-UL2018_MiniAODv2-v2/out_738.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/110000/DC468440-CA82-5D4E-85E7-1D046CBF8054.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018B-UL2018_MiniAODv2-v2/out_1030.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/11FC6FFD-193E-9641-8F1B-C8C9B6A03553.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018B-UL2018_MiniAODv2-v2/out_505.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/7FA66E50-0576-9E4E-8532-A76B5B900E04.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018B-UL2018_MiniAODv2-v2/out_502.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/0E6991E5-FFC1-6C45-8309-CF7B28053F8B.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018B-UL2018_MiniAODv2-v2/out_503.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/BDA6415C-64CA-654F-AF96-C963AA217B78.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018B-UL2018_MiniAODv2-v2/out_500.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/CBAFC35B-F7C5-FD4C-9E2D-D3B0A984C9C9.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018B-UL2018_MiniAODv2-v2/out_501.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/8FFE9544-00D9-FC44-AB5D-A788BF56F35E.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018B-UL2018_MiniAODv2-v2/out_630.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/0501E152-9B65-D44E-8158-257F2D9ABB38.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018B-UL2018_MiniAODv2-v2/out_631.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/A3D326D3-4BF4-4B42-93E5-A0288E6257E2.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018B-UL2018_MiniAODv2-v2/out_632.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/0BCECABB-DC7D-3740-8CE8-9A06A02E3EF2.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018B-UL2018_MiniAODv2-v2/out_633.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/3A321BB4-7602-C643-8546-9B3576ED0F23.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018B-UL2018_MiniAODv2-v2/out_469.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/718AEF93-B04A-DB4F-9076-8BD0B5D311E5.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018B-UL2018_MiniAODv2-v2/out_468.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/D39BB452-6ADF-DC44-812E-C998698151C9.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018B-UL2018_MiniAODv2-v2/out_636.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/F1582CF6-CD30-EA46-A04F-7CF9E6C4B17F.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018B-UL2018_MiniAODv2-v2/out_637.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/F2538364-262E-BE4E-90C6-24B0C10EBA8B.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018B-UL2018_MiniAODv2-v2/out_465.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/814EB154-1769-CB4F-924F-DBD38C93AA98.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018B-UL2018_MiniAODv2-v2/out_464.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/A1472696-6F60-9E43-9F8F-9AE28D1927BF.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018B-UL2018_MiniAODv2-v2/out_467.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/B17D06C7-F7CA-0449-AAF2-B38E00447105.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018B-UL2018_MiniAODv2-v2/out_466.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/90A1C6C0-8D0E-844D-85C7-708C1EF60B23.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018B-UL2018_MiniAODv2-v2/out_461.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/110000/A4E6FE26-17DD-9843-A7FA-85C3092C15B4.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018B-UL2018_MiniAODv2-v2/out_900.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/F90EE915-F8FE-C049-ACE1-C81EA0669D54.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018B-UL2018_MiniAODv2-v2/out_463.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/63732270-A2D1-3047-8E54-1910AFE8ECD3.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018B-UL2018_MiniAODv2-v2/out_462.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/110000/45B41A52-C327-F540-AE27-E5A885887B84.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018B-UL2018_MiniAODv2-v2/out_901.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/270000/0277F7BF-7E41-DD44-B4DB-D771108E0E01.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018B-UL2018_MiniAODv2-v2/out_168.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/270000/940798A7-4D3C-3040-ABFD-C5022A58C024.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018B-UL2018_MiniAODv2-v2/out_169.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/270000/DBB9DB63-209D-0741-9B12-D96DD8BD7EB7.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018B-UL2018_MiniAODv2-v2/out_164.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/270000/5927C01C-E0E9-7341-8AEC-FF90A8E74B48.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018B-UL2018_MiniAODv2-v2/out_165.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/270000/F32422F6-0227-574E-8159-0A0F0E504A59.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018B-UL2018_MiniAODv2-v2/out_166.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/270000/508C816C-7A8A-F449-B37F-B69FC5CD2B86.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018B-UL2018_MiniAODv2-v2/out_167.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/270000/5FFBB067-35F1-DC49-8B24-E84CEB6298D4.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018B-UL2018_MiniAODv2-v2/out_160.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/270000/DB6B50C7-D2FB-7C42-8D4B-5895C4D7AA86.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018B-UL2018_MiniAODv2-v2/out_161.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/270000/71920654-D22F-FC41-8343-93AF7A029A31.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018B-UL2018_MiniAODv2-v2/out_162.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/270000/76C199F2-570F-2249-BBAF-35B3DCCDDECD.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018B-UL2018_MiniAODv2-v2/out_163.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/9D5C797E-62DC-784F-82A6-C9757EA04D6B.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018B-UL2018_MiniAODv2-v2/out_964.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/A6476FD3-966D-8641-8F97-F548983811BD.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018B-UL2018_MiniAODv2-v2/out_965.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/20EDE70B-ECA2-7942-9E56-45B5BF7AEB1A.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018B-UL2018_MiniAODv2-v2/out_966.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/A4367418-2A03-4545-B2C5-2C9B8A2EAF50.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018B-UL2018_MiniAODv2-v2/out_967.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/09FB91C7-2B5E-4D48-9A94-19A2E0981EDC.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018B-UL2018_MiniAODv2-v2/out_960.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/A7807416-8EFD-6448-A2BF-E4BB04679D5B.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018B-UL2018_MiniAODv2-v2/out_961.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/3D91F4CC-AF51-624B-B92B-635B4001DA5D.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018B-UL2018_MiniAODv2-v2/out_962.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/CC2DA1BD-3771-0947-AEC2-16CAD19724BD.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018B-UL2018_MiniAODv2-v2/out_963.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/110000/042DDFDC-B3D5-CD49-97FF-BDBFDBC66626.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018B-UL2018_MiniAODv2-v2/out_1039.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/110000/89259D20-DC29-3048-B66B-94F43746194F.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018B-UL2018_MiniAODv2-v2/out_783.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/463ED7BC-097D-3547-A74D-E4AC9B4BA478.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018B-UL2018_MiniAODv2-v2/out_968.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/898E3CD8-4C80-7E46-AF2A-CDAF82D0C096.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018B-UL2018_MiniAODv2-v2/out_969.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/110000/BA57F884-2A1A-6043-B710-6A0F30147D8A.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018B-UL2018_MiniAODv2-v2/out_936.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/674E60F2-BA0C-3342-9E8B-BBC8EF24C4A0.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018B-UL2018_MiniAODv2-v2/out_1106.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/8E18DFD3-9374-B048-9259-6CD760D2F46F.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018B-UL2018_MiniAODv2-v2/out_1107.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/22E87915-B006-0A48-9AD7-83A2BD43FD1F.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018B-UL2018_MiniAODv2-v2/out_1104.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/74A0811A-B55E-B942-9C1B-90F18BCC0273.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018B-UL2018_MiniAODv2-v2/out_1105.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/4CD47DB8-55EF-6A4E-A8DE-1EEA22913FEE.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018B-UL2018_MiniAODv2-v2/out_1102.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/A51C95A5-D262-7B4D-8A0F-8E0AF61DC5F7.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018B-UL2018_MiniAODv2-v2/out_1103.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/8AC5D376-499D-1B4E-88C3-9C980814200E.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018B-UL2018_MiniAODv2-v2/out_1100.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/B0F5F1C8-2A29-B24D-A969-6C443F982A4B.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018B-UL2018_MiniAODv2-v2/out_1101.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/110000/87AA1FAC-E009-D84C-87AE-F4F1B18C2155.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018B-UL2018_MiniAODv2-v2/out_934.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/110000/608AB98B-3965-584D-8721-F60D8101B0E6.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018B-UL2018_MiniAODv2-v2/out_908.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/51B79D43-AFCD-BD48-922F-AFA4690B5AD1.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018B-UL2018_MiniAODv2-v2/out_1108.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/40000/7EFA3785-3926-CE45-B59B-22B262BBCE6A.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018B-UL2018_MiniAODv2-v2/out_1109.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/110000/D0D111FE-EE37-2748-AD0C-E08BC002CCA5.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018B-UL2018_MiniAODv2-v2/out_909.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/4A257FD2-9FDF-A247-9416-7BDF1BAB88E6.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018B-UL2018_MiniAODv2-v2/out_722.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/68766C38-C18E-3A4F-998A-49E1EFFDE80A.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018B-UL2018_MiniAODv2-v2/out_678.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/42926619-39CD-7F4A-B319-13E78EF48408.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018B-UL2018_MiniAODv2-v2/out_679.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/71DB0411-52A7-1A42-9CE7-E0DC434EF41A.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018B-UL2018_MiniAODv2-v2/out_426.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/110000/EF51BB4E-C8EB-4E46-A0F2-03194A45180B.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018B-UL2018_MiniAODv2-v2/out_878.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/110000/A2BCACA6-6596-BB45-BD85-FA3D62AD074D.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018B-UL2018_MiniAODv2-v2/out_879.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/110000/84F31A7F-4386-E042-AFF9-BFD8DC9031A3.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018B-UL2018_MiniAODv2-v2/out_876.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/110000/6526EABF-FFF1-C942-9FCA-B25BF3BA3556.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018B-UL2018_MiniAODv2-v2/out_877.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/56BDF931-6CF9-9145-B321-1EDF33F1CA68.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018B-UL2018_MiniAODv2-v2/out_874.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/110000/E38E5C1B-F994-BF48-B4DC-0D7F26234BCC.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018B-UL2018_MiniAODv2-v2/out_875.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/06C1D238-9D15-CB44-9C98-275A61EFBCDA.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018B-UL2018_MiniAODv2-v2/out_872.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/7A4F4C04-B7A3-7441-90A4-619707868531.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018B-UL2018_MiniAODv2-v2/out_873.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/A9AA5575-843F-6F4B-A3CF-3D671B1F4223.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018B-UL2018_MiniAODv2-v2/out_870.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/BEC93036-F97D-3747-8FE6-1D8504BBD629.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018B-UL2018_MiniAODv2-v2/out_871.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/4A9652CE-F036-BC4F-8C12-ACD65E456151.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018B-UL2018_MiniAODv2-v2/out_723.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/260000/2CD3B14F-0893-8E4E-92B0-D1497D72833C.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018B-UL2018_MiniAODv2-v2/out_9.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/110000/E157AAB1-049E-7C4A-A08C-CEB25D622DAC.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018B-UL2018_MiniAODv2-v2/out_890.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/110000/37FFD504-F039-FD43-9785-494C4EEB5160.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018B-UL2018_MiniAODv2-v2/out_891.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/110000/5EA265CD-A064-D14A-BA1C-8FC50C02290A.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018B-UL2018_MiniAODv2-v2/out_892.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/110000/8659F8E2-0407-3E4B-9C5A-1EAB7BF08A70.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018B-UL2018_MiniAODv2-v2/out_893.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/110000/EA19DB5A-E2DE-0A44-89BC-12C93473C502.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018B-UL2018_MiniAODv2-v2/out_894.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/C5FA0CA6-FDB9-D14C-9FDF-81922692C3DB.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018B-UL2018_MiniAODv2-v2/out_647.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/110000/FEB4DAEA-F500-5945-86B8-A4338CB59CD4.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018B-UL2018_MiniAODv2-v2/out_896.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/110000/97B980CD-89D4-5E44-98D5-311EC1580AF3.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018B-UL2018_MiniAODv2-v2/out_897.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/110000/18DE2503-FE97-D644-9221-243FB0222120.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018B-UL2018_MiniAODv2-v2/out_898.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/110000/C6ECE16F-D6F1-AB4B-A52B-3C1CE069666E.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018B-UL2018_MiniAODv2-v2/out_899.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/1D081D89-B41D-AF40-A0AE-DA73A4C8430A.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018B-UL2018_MiniAODv2-v2/out_1087.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/3051EED0-9510-5147-A53A-3441D3CEC271.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018B-UL2018_MiniAODv2-v2/out_646.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/DD617A79-1466-2C4C-9AC2-FC57B771640F.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018B-UL2018_MiniAODv2-v2/out_671.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/1AB42A0C-5943-FF4C-9165-B323490667AA.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018B-UL2018_MiniAODv2-v2/out_649.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/0B6F4F30-D0EE-2C4D-A2CC-DFA1F53ED639.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018B-UL2018_MiniAODv2-v2/out_648.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/A8916A45-5339-A542-BBC2-144538F1ADC8.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018B-UL2018_MiniAODv2-v2/out_1086.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/1EC87D53-EF52-1340-8DA0-A8289A07EC15.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018B-UL2018_MiniAODv2-v2/out_357.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/E5A60193-6D68-794A-B9F4-5A2AA4E4B1F0.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018B-UL2018_MiniAODv2-v2/out_356.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/110000/DF1C4085-BF0F-804F-ADBC-A3C5CC2891FA.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018B-UL2018_MiniAODv2-v2/out_809.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/110000/012BD380-53E6-4B4E-802B-3F96BF7DA0E5.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018B-UL2018_MiniAODv2-v2/out_808.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/7CCB7F6F-7AF0-7C4D-B7B1-DCABCE247343.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018B-UL2018_MiniAODv2-v2/out_353.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/3F814745-0FE4-BC46-8792-D965C11AE421.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018B-UL2018_MiniAODv2-v2/out_352.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/C8215A5E-7176-594B-B175-89957641553F.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018B-UL2018_MiniAODv2-v2/out_351.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/930FE4F7-0998-A34A-B4FD-CDEDAF34B058.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018B-UL2018_MiniAODv2-v2/out_350.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/110000/797ABC73-800D-804D-954B-6508BD15BB3D.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018B-UL2018_MiniAODv2-v2/out_803.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/110000/7A0D691F-B3C8-094C-8B0A-093FD84CB408.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018B-UL2018_MiniAODv2-v2/out_802.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/110000/ECE5DB69-0BC7-D44F-8F5D-E8FC2D6B6801.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018B-UL2018_MiniAODv2-v2/out_801.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/110000/B169C25E-062C-8E4C-A5DB-619B5F260551.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018B-UL2018_MiniAODv2-v2/out_800.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/110000/8F08BACC-5E79-F243-8BFF-9ACE3A184A7F.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018B-UL2018_MiniAODv2-v2/out_807.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/110000/3A8F7A2E-D4D6-BC45-AF57-30D1F160629C.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018B-UL2018_MiniAODv2-v2/out_806.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/110000/6AEE9570-6EFF-6B4E-9A63-B1862F89C32F.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018B-UL2018_MiniAODv2-v2/out_805.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/110000/FFEDC76D-B88B-8B40-BA50-20ED34D94B57.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018B-UL2018_MiniAODv2-v2/out_804.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/12E74ADE-60B7-6144-9827-772C2C170C4D.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018B-UL2018_MiniAODv2-v2/out_216.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/DCBCD5EC-82FA-144C-9C86-BC1E2DD6D02B.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018B-UL2018_MiniAODv2-v2/out_217.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/110000/EBC6F20B-822E-FA40-AC55-814B82796B7A.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018B-UL2018_MiniAODv2-v2/out_768.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/110000/B8F7A891-174B-A142-AD93-074DC9B0DC70.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018B-UL2018_MiniAODv2-v2/out_769.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/68E6279F-9010-0240-98D5-160434BD8254.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018B-UL2018_MiniAODv2-v2/out_212.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/D14A227C-A8FE-4047-B681-9DBFDB5F5795.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018B-UL2018_MiniAODv2-v2/out_213.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/100000/B29BDCC8-5429-7244-96D9-A5BF6C04BB73.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018B-UL2018_MiniAODv2-v2/out_210.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/A3E6A4F5-BFC6-644C-8471-204546779E5E.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018B-UL2018_MiniAODv2-v2/out_211.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/110000/A0B25263-88F4-2C4A-8CF5-11F76329468F.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018B-UL2018_MiniAODv2-v2/out_762.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/110000/C55A05C7-EE2C-D246-8BEC-26CD3FF3D7B3.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018B-UL2018_MiniAODv2-v2/out_763.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/110000/6711E8A1-027B-2D44-9CBF-92941D239FBD.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018B-UL2018_MiniAODv2-v2/out_760.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/110000/B63147A7-B4CC-BE49-8DA2-DF8DFF6A2AB4.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018B-UL2018_MiniAODv2-v2/out_761.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/110000/D3C15F31-BEFB-7841-9D44-D1FB9B14D113.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018B-UL2018_MiniAODv2-v2/out_766.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/110000/A62D9B33-026C-C34E-B5AE-9216FB3B06ED.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018B-UL2018_MiniAODv2-v2/out_767.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/110000/3126D576-7EE7-2A42-B536-FB5FB27BB4F9.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018B-UL2018_MiniAODv2-v2/out_764.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/110000/3F8D78B0-7019-E54A-AAC8-3E7D44602CB5.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018B-UL2018_MiniAODv2-v2/out_765.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/260000/B4642635-0BAB-1348-9A39-07B175049583.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018B-UL2018_MiniAODv2-v2/out_40.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/260000/D57E7614-F3D9-6D4D-A134-3FA49E5ABD3B.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018B-UL2018_MiniAODv2-v2/out_41.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/110000/94673D53-EF31-7C4F-ABC1-23066FB2A8A3.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018B-UL2018_MiniAODv2-v2/out_1033.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/7C905C48-B6CD-3A4B-AE2B-93504020327C.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018B-UL2018_MiniAODv2-v2/out_1078.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/81140244-7D75-5741-BF6C-D0CFB35A8084.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018B-UL2018_MiniAODv2-v2/out_1079.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/E8772087-214D-1342-9CE7-360D47BD0489.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018B-UL2018_MiniAODv2-v2/out_1076.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/20378EE5-4D8F-7846-AE51-22E251751A7C.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018B-UL2018_MiniAODv2-v2/out_1077.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/056A03A4-F3F3-5C49-AB07-56EAD7A66D56.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018B-UL2018_MiniAODv2-v2/out_1074.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/E7B414FE-89AE-3042-8A54-BE108A8D3624.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018B-UL2018_MiniAODv2-v2/out_1075.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/4EDC5E3A-57BE-3046-92FC-8891C7BBD1DF.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018B-UL2018_MiniAODv2-v2/out_1072.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/052DFA6D-E502-0849-8AAE-DC02FF85783A.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018B-UL2018_MiniAODv2-v2/out_1073.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/8D442703-B31B-E14D-8BF5-95AA72E14843.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018B-UL2018_MiniAODv2-v2/out_1070.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/B4381177-3CC2-CE40-8766-C0F954F7CC5B.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018B-UL2018_MiniAODv2-v2/out_1071.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/30BDF644-AA16-B840-9614-1B19A916BBA0.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018B-UL2018_MiniAODv2-v2/out_289.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/54134162-F374-4943-BF53-24D04CC2A0FA.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018B-UL2018_MiniAODv2-v2/out_288.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/110000/0E0B09C5-D9CE-3945-A2E7-7E42BE901AC3.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018B-UL2018_MiniAODv2-v2/out_1003.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/260000/99DE6755-49CC-AE45-8AD2-C3702CED0974.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018B-UL2018_MiniAODv2-v2/out_4.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/3A8D1CEC-F979-1C4E-AC12-A2117872B324.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018B-UL2018_MiniAODv2-v2/out_281.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/87096E45-E059-064B-80EA-264A3D81BE7E.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018B-UL2018_MiniAODv2-v2/out_280.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/AD3CC22A-5E2D-5A47-B74F-9E262F028427.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018B-UL2018_MiniAODv2-v2/out_283.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/F9D6232D-9673-B443-A875-00FFB7A77C1B.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018B-UL2018_MiniAODv2-v2/out_282.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/69CF51AB-801A-8D4A-B7E3-DAFBB88D0253.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018B-UL2018_MiniAODv2-v2/out_285.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/883135FC-7930-7E49-8CEE-3EDF03E70BA6.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018B-UL2018_MiniAODv2-v2/out_284.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/BD25E312-EC3A-C541-9400-A5A0FB78B9FF.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018B-UL2018_MiniAODv2-v2/out_287.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/9F87F694-C0FB-8E46-8B61-CA8A77DF0168.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018B-UL2018_MiniAODv2-v2/out_286.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/0EB6DA6D-69E5-FB4F-8B61-2C37398161F5.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018B-UL2018_MiniAODv2-v2/out_1094.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/30E86BFD-80A7-F14E-99F1-957546D75A4A.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018B-UL2018_MiniAODv2-v2/out_1095.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/171CE9BC-E84C-284D-B16E-6451F9F3DDF5.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018B-UL2018_MiniAODv2-v2/out_1096.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/8F44334D-0149-5F4B-B05B-A259E79FD031.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018B-UL2018_MiniAODv2-v2/out_1097.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/195ADDB4-5973-5F40-AF77-FB240FA7D694.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018B-UL2018_MiniAODv2-v2/out_1090.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/44FF20FB-97EA-0047-AFF7-9E49FEAAD758.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018B-UL2018_MiniAODv2-v2/out_1091.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/1FB7B62C-77BC-F947-B7A6-FF7D314D3FF7.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018B-UL2018_MiniAODv2-v2/out_1092.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/FFFC6C4E-DE67-4C4E-A11B-F0F4FE858361.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018B-UL2018_MiniAODv2-v2/out_1093.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/BB6ACC17-0FB8-954F-94F4-36691ACBCBB3.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018B-UL2018_MiniAODv2-v2/out_674.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/43D21203-88E5-BC40-B4B1-8CCE2C663FD9.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018B-UL2018_MiniAODv2-v2/out_675.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/D2CA12A5-C317-0441-9FBA-C04967CDB1A1.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018B-UL2018_MiniAODv2-v2/out_676.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/FD554419-29AC-F844-A739-28878298BE91.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018B-UL2018_MiniAODv2-v2/out_677.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/44603034-448B-1145-84E3-9936A00AE9EE.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018B-UL2018_MiniAODv2-v2/out_1098.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/9C269974-0DB5-0546-940F-D33B8470D12E.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018B-UL2018_MiniAODv2-v2/out_1099.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/FE8F35DB-CCE6-164B-BD4A-2A676B3ED3E6.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018B-UL2018_MiniAODv2-v2/out_672.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/3B322327-8AF1-554E-BE31-E268696D644A.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018B-UL2018_MiniAODv2-v2/out_673.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/96BA0A44-2CB9-B345-B7AC-73F5A1AB0E4A.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018B-UL2018_MiniAODv2-v2/out_263.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/43CD80C9-01D3-0D43-A925-E6B98E3A8A62.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018B-UL2018_MiniAODv2-v2/out_262.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/9B46D2E9-54B3-AA4A-A928-2F6980053C92.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018B-UL2018_MiniAODv2-v2/out_261.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/FE6331AA-0093-DD4D-9533-58BB0543E0C2.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018B-UL2018_MiniAODv2-v2/out_260.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/6C20D40D-2318-CF4A-A888-9985EDCE4CF6.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018B-UL2018_MiniAODv2-v2/out_267.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/B12C8398-0B08-2F47-9ABA-9BE6F758ECB2.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018B-UL2018_MiniAODv2-v2/out_266.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/9495416B-ECD4-4E48-8DF6-B81A6EF0B77D.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018B-UL2018_MiniAODv2-v2/out_265.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/05AA289B-B36A-F84D-8079-B8091CF63A4C.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018B-UL2018_MiniAODv2-v2/out_264.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/8CC6AA36-B66C-8847-9E55-F92F35CFF287.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018B-UL2018_MiniAODv2-v2/out_269.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/8D2D178A-8D9C-6842-A270-8EE04E4B19D7.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018B-UL2018_MiniAODv2-v2/out_268.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/DE9E8560-B074-7949-8044-8C361F52BB11.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018B-UL2018_MiniAODv2-v2/out_478.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/260000/BF1F7852-8C82-C34E-A25A-978BF759FE57.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018B-UL2018_MiniAODv2-v2/out_59.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/260000/83056DC5-1415-6F4D-A1D1-AC61E5C77A27.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018B-UL2018_MiniAODv2-v2/out_58.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/260000/58ED18F2-B68C-F844-985E-A4A2BF8D9ED4.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018B-UL2018_MiniAODv2-v2/out_55.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/260000/3D52BA6F-6FA8-5B44-98D5-BB3A568BC182.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018B-UL2018_MiniAODv2-v2/out_54.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/260000/ABAC4A4B-3648-7D47-9C53-2B7B935D836D.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018B-UL2018_MiniAODv2-v2/out_57.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/260000/ED740900-134E-3540-9C7F-A4459F7612F6.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018B-UL2018_MiniAODv2-v2/out_56.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/260000/389EB5C7-732E-9C46-A408-3C347512F596.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018B-UL2018_MiniAODv2-v2/out_51.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/E4FD8A22-5932-884D-BABF-C47BD98368BC.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018B-UL2018_MiniAODv2-v2/out_258.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/260000/7ABE196A-3B12-984E-82A4-24E59FDDE407.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018B-UL2018_MiniAODv2-v2/out_53.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/260000/271056C9-58B2-664E-8FEA-90E4821A9E56.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018B-UL2018_MiniAODv2-v2/out_52.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/70021736-676F-C547-A68E-16B2435E7274.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018B-UL2018_MiniAODv2-v2/out_537.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/EB586717-8162-F84D-8209-BE20D9163EF0.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018B-UL2018_MiniAODv2-v2/out_536.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/07802F4E-D05F-D045-808E-3E2600E928C2.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018B-UL2018_MiniAODv2-v2/out_535.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/260000/7C53F3D2-D56C-7245-8F2B-C37FBA83C3CD.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018B-UL2018_MiniAODv2-v2/out_63.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/ABA75B0D-1B7B-C545-9241-8CE29D042780.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018B-UL2018_MiniAODv2-v2/out_533.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/656B19CC-3C7E-B44B-B714-647DE93FD985.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018B-UL2018_MiniAODv2-v2/out_532.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/1C205F02-3F07-CB4F-9A6F-319B9C999BE2.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018B-UL2018_MiniAODv2-v2/out_531.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/5E49C152-7186-AD4A-907C-0FA1C87FEC82.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018B-UL2018_MiniAODv2-v2/out_530.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/E6A8FF65-0484-3B4C-8FDD-B650BE736F3D.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018B-UL2018_MiniAODv2-v2/out_598.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/231CACE7-D8EE-784A-A18D-134EF266C117.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018B-UL2018_MiniAODv2-v2/out_152.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/B3DF8D10-0DBF-BE43-87EC-201A623B8BD8.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018B-UL2018_MiniAODv2-v2/out_539.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/5AA53D4A-CA06-9348-9808-26CFA0D52BC1.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018B-UL2018_MiniAODv2-v2/out_538.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/110000/989C51EA-3243-B446-A39A-1E3748190981.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018B-UL2018_MiniAODv2-v2/out_775.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/110000/30A3BAF0-3AA8-2443-8F6D-5078EE77B3F2.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018B-UL2018_MiniAODv2-v2/out_988.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/110000/69EC40A0-5A11-774F-A3BA-42B140A47C4D.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018B-UL2018_MiniAODv2-v2/out_989.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/110000/6231E232-D747-4B4F-9699-371E9F699D1C.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018B-UL2018_MiniAODv2-v2/out_774.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/94CDF829-D16C-9643-8DAE-854087C57EBD.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018B-UL2018_MiniAODv2-v2/out_982.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/4B27C312-88D7-2442-A70A-17FEB6A3ACA6.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018B-UL2018_MiniAODv2-v2/out_983.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/F53FEA9F-D80D-A746-BCF1-53807DFC1CD7.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018B-UL2018_MiniAODv2-v2/out_980.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/0DA11DD7-6806-7148-B535-75271E0B666E.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018B-UL2018_MiniAODv2-v2/out_981.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/110000/22FD4FC7-2243-7D44-952D-B4CD95305BA3.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018B-UL2018_MiniAODv2-v2/out_986.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/110000/6164755F-B6D3-2346-9EEB-943705790357.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018B-UL2018_MiniAODv2-v2/out_987.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/110000/2519D7D3-B0BA-DE46-8FE4-332C838232FA.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018B-UL2018_MiniAODv2-v2/out_984.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/260000/013048C9-15DA-AB4B-BC97-AB7A8658C794.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018B-UL2018_MiniAODv2-v2/out_50.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/260000/6C69236F-EB78-A343-9928-D2BB5A320788.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018B-UL2018_MiniAODv2-v2/out_115.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/4D93D066-CE5B-6446-A3EC-DFA3CF428267.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018B-UL2018_MiniAODv2-v2/out_252.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/260000/1D422108-CF60-E546-92CC-DD7531A7DFCE.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018B-UL2018_MiniAODv2-v2/out_117.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/110000/4A469BD6-F91F-C842-8055-0839FCDFB471.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018B-UL2018_MiniAODv2-v2/out_776.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/260000/0F2184DD-5DAA-7241-A790-DEA0DCBA54AB.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018B-UL2018_MiniAODv2-v2/out_111.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/260000/7C6B5E6F-57E0-884E-8F78-193284F6B7AB.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018B-UL2018_MiniAODv2-v2/out_110.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/260000/17232B9B-EEDB-6F4F-9A18-027B268DE7FC.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018B-UL2018_MiniAODv2-v2/out_113.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/260000/F505C1B5-E900-6644-8E37-E63444E55144.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018B-UL2018_MiniAODv2-v2/out_69.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/110000/164DF4EA-E8B8-5344-B655-EDB88F5DE244.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018B-UL2018_MiniAODv2-v2/out_771.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/0628AC3F-2BF1-6547-A19A-F77EEDB20FE2.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018B-UL2018_MiniAODv2-v2/out_119.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/11C4A6C3-4A82-6A48-B3ED-ED0DA86856A2.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018B-UL2018_MiniAODv2-v2/out_118.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/110000/87648260-7875-754E-B0B9-9CD30C147701.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018B-UL2018_MiniAODv2-v2/out_770.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/110000/9B5AE0B9-8C69-1049-BC2C-0F508549F232.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018B-UL2018_MiniAODv2-v2/out_773.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/110000/B881C5DA-DA24-7445-8431-E6278182037C.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018B-UL2018_MiniAODv2-v2/out_772.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/110000/77B8250B-BB36-D442-A524-F198AC6914CE.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018B-UL2018_MiniAODv2-v2/out_1001.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/76513E63-5DD6-664D-B694-B4D8DF705CFE.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018B-UL2018_MiniAODv2-v2/out_429.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/E11341C3-D757-654F-BAE3-F35C03D15718.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018B-UL2018_MiniAODv2-v2/out_428.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/EED0A73D-B416-A94B-8373-8CB0D5F51517.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018B-UL2018_MiniAODv2-v2/out_534.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/110000/B1BE4E74-FA02-3047-ADA9-6CE0152D6159.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018B-UL2018_MiniAODv2-v2/out_919.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/110000/597BBCD0-6771-6C41-9849-47FB70F89DD6.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018B-UL2018_MiniAODv2-v2/out_918.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/110000/919598E1-D3AA-9A40-BE23-5682E4F2C40A.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018B-UL2018_MiniAODv2-v2/out_915.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/110000/D62106B0-1468-1D48-9521-7675701520F6.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018B-UL2018_MiniAODv2-v2/out_914.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/110000/B616C456-D543-BE43-B904-664581B2A0F9.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018B-UL2018_MiniAODv2-v2/out_917.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/110000/EEA43AF3-2BCD-C540-8B12-20BD25C780D3.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018B-UL2018_MiniAODv2-v2/out_916.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/110000/DABEA637-C5B8-144A-BE0B-D6EF1E2452C1.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018B-UL2018_MiniAODv2-v2/out_911.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/110000/E16AD36E-04C1-6147-9214-F7A253273603.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018B-UL2018_MiniAODv2-v2/out_910.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/110000/A00941A5-363A-BB4C-8B6E-2563F5559BE9.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018B-UL2018_MiniAODv2-v2/out_913.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/110000/915892BE-2A58-9246-A0D7-420431635F47.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018B-UL2018_MiniAODv2-v2/out_912.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/1FE82AF5-C4AB-4A46-B94A-673328EC34E9.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018B-UL2018_MiniAODv2-v2/out_308.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/321705B9-8DBB-984D-AFDC-945705A84DC5.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018B-UL2018_MiniAODv2-v2/out_309.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/110000/41B94AC4-32CB-A346-9D95-DFC0AF4DF31F.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018B-UL2018_MiniAODv2-v2/out_855.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/5438C97B-E549-D740-B368-08EEAC174424.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018B-UL2018_MiniAODv2-v2/out_300.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/D257D93D-ED84-C24E-BC5C-E6CF281E8B55.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018B-UL2018_MiniAODv2-v2/out_301.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/4C7CB693-2FE8-A741-A04A-0C13F4D6DABD.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018B-UL2018_MiniAODv2-v2/out_302.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/8D0D187C-FA4C-0647-96D2-F99D24CE6C17.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018B-UL2018_MiniAODv2-v2/out_303.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/58C87210-D580-694D-B3C5-18971DA71211.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018B-UL2018_MiniAODv2-v2/out_304.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/FBA88B23-AF2F-2F41-989C-B8261A87E0F9.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018B-UL2018_MiniAODv2-v2/out_305.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/972FBC5F-1BCB-4B46-BAEB-B4E287FA8BA6.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018B-UL2018_MiniAODv2-v2/out_306.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/021F87F2-309F-4549-9DE7-377BA407321D.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018B-UL2018_MiniAODv2-v2/out_307.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/110000/DF6738E6-BA5C-F049-84E7-C136DBFE729A.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018B-UL2018_MiniAODv2-v2/out_895.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/110000/E9CBD81A-8994-9849-8432-F85D1E82065C.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018B-UL2018_MiniAODv2-v2/out_825.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/110000/80722CD8-35D8-3841-A613-EB5E7B8D9A29.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018B-UL2018_MiniAODv2-v2/out_824.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/110000/13AEC7E0-042A-2849-8FF5-EB2C334E29CE.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018B-UL2018_MiniAODv2-v2/out_827.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/110000/1CA38B69-4B43-CA41-9118-F973FC1DC585.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018B-UL2018_MiniAODv2-v2/out_847.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/110000/65EEC8BB-E437-B640-AFF9-2E70DA44FA02.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018B-UL2018_MiniAODv2-v2/out_846.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/110000/8B584612-9484-E645-84E4-D5A510959A64.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018B-UL2018_MiniAODv2-v2/out_845.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/110000/232F1F55-CFCA-9A46-8447-D51381A9BDB6.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018B-UL2018_MiniAODv2-v2/out_844.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/110000/E48E456E-07B0-1D43-B804-3DE73100D496.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018B-UL2018_MiniAODv2-v2/out_843.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/19EE95D9-93E9-2848-B97A-60306F26B2C6.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018B-UL2018_MiniAODv2-v2/out_842.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/A592B869-9020-8249-915E-B49D86E2F176.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018B-UL2018_MiniAODv2-v2/out_841.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/3638CF78-A8F9-3947-8332-7E9A2D155F9D.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018B-UL2018_MiniAODv2-v2/out_840.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/110000/744DB590-61A3-8745-BB19-FB74DD14BFBC.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018B-UL2018_MiniAODv2-v2/out_821.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/110000/B5DB46D9-96C4-4D43-8E2C-4BF169F48C01.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018B-UL2018_MiniAODv2-v2/out_853.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/110000/51C2599F-3C8D-274E-8ACB-BCC999621421.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018B-UL2018_MiniAODv2-v2/out_849.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/110000/0BAF8A35-7FF2-204B-A05A-CEE1EEFC3AA1.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018B-UL2018_MiniAODv2-v2/out_848.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/110000/E9B6D915-A2EB-684F-8043-25BB20EC56A7.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018B-UL2018_MiniAODv2-v2/out_823.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/110000/D4AE3508-2680-324A-943F-DB3E6697CE76.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018B-UL2018_MiniAODv2-v2/out_822.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/FE370C8D-0D2B-574E-A08B-9481DCF2F553.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018B-UL2018_MiniAODv2-v2/out_240.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/4F387617-0417-194C-A5AD-6F0DDCB26289.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018B-UL2018_MiniAODv2-v2/out_485.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/CDDCA6B1-3A55-6244-B5F3-403307B2AFC8.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018B-UL2018_MiniAODv2-v2/out_390.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/110000/EDBB1966-F3BB-3345-BC0A-A59C972A1401.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018B-UL2018_MiniAODv2-v2/out_1038.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/110000/2A95D7AF-51A8-424F-85E4-7EC54D056FAB.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018B-UL2018_MiniAODv2-v2/out_732.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/C004E7E7-DFE4-2542-AE90-25E1CB1FFD4B.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018B-UL2018_MiniAODv2-v2/out_391.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/110000/469C8816-2315-D841-84A3-C924CA4D19C3.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018B-UL2018_MiniAODv2-v2/out_753.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/110000/952F0443-760E-A349-AB24-A819100C1A01.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018B-UL2018_MiniAODv2-v2/out_752.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/110000/0C26BF16-D3DF-9640-9823-974EAD9A59B4.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018B-UL2018_MiniAODv2-v2/out_751.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/110000/FCE7B98C-7460-244A-BF8C-DDD39328772B.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018B-UL2018_MiniAODv2-v2/out_750.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/110000/9948045F-CC86-8543-A33F-822C7671F04E.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018B-UL2018_MiniAODv2-v2/out_757.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/110000/F4E6A6EC-11E8-944B-BD10-EE261667284F.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018B-UL2018_MiniAODv2-v2/out_756.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/110000/4949E8F6-BF3F-9644-B6CC-B32B1136941E.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018B-UL2018_MiniAODv2-v2/out_755.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/110000/BF6D8FCA-5F0C-504B-848D-70C143FA09E3.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018B-UL2018_MiniAODv2-v2/out_754.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/E6936008-4532-C148-99EF-62276005AACB.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018B-UL2018_MiniAODv2-v2/out_560.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/962D5CBB-51EF-ED4A-BF04-95E012F84C9B.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018B-UL2018_MiniAODv2-v2/out_561.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/110000/063E97DF-59E2-8448-90D4-81B38D246654.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018B-UL2018_MiniAODv2-v2/out_759.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/110000/B5512B6B-1BBA-9D49-BFC6-0915677066E8.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018B-UL2018_MiniAODv2-v2/out_758.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/37126B60-4D3F-D542-B7B5-8E2AFDDA7E56.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018B-UL2018_MiniAODv2-v2/out_564.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/8A73FF66-6342-3B40-83B2-AE327845EACF.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018B-UL2018_MiniAODv2-v2/out_565.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/30B0F11D-C5C2-6143-BE31-FD43489ABEF9.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018B-UL2018_MiniAODv2-v2/out_566.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/6CF2B496-C3CE-3749-8CA8-7E43D4166DDC.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018B-UL2018_MiniAODv2-v2/out_701.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/DF44333E-D589-D140-B335-6E85480B5FD7.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018B-UL2018_MiniAODv2-v2/out_506.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/F66C5505-8B84-3949-BE92-C97DEDE0D147.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018B-UL2018_MiniAODv2-v2/out_229.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/629DB64A-9ADD-0546-9EF8-AE29AA19CE34.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018B-UL2018_MiniAODv2-v2/out_507.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/84BADB21-82F9-BF43-93F1-6F10A73D364F.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018B-UL2018_MiniAODv2-v2/out_227.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/129BC45C-78F5-6043-A5C6-5DCD7A3B5E58.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018B-UL2018_MiniAODv2-v2/out_226.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/DD3B5F1F-4ABF-854B-886D-37814FB4F33A.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018B-UL2018_MiniAODv2-v2/out_225.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/110000/22E4A561-AB5A-B147-A9C5-72B66989314F.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018B-UL2018_MiniAODv2-v2/out_985.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/74E6142C-C6AD-304E-901D-16384B54121C.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018B-UL2018_MiniAODv2-v2/out_223.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/45E677EC-A5CC-E04B-9C51-ED69E00AA88D.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018B-UL2018_MiniAODv2-v2/out_504.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/B3C4A8EF-3BC3-0940-94E8-8816B0DDDBAA.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018B-UL2018_MiniAODv2-v2/out_221.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/F6B785CA-B4CF-8D49-BB95-B02FD9E8412C.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018B-UL2018_MiniAODv2-v2/out_220.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/110000/9424DE59-8593-004D-81AF-D92A1BDE709F.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018B-UL2018_MiniAODv2-v2/out_1025.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/110000/3A70C00E-DBF2-594A-8973-0D47EB98DD3F.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018B-UL2018_MiniAODv2-v2/out_1024.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/110000/639AA11D-48CF-8342-9D42-F88029E4EB6F.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018B-UL2018_MiniAODv2-v2/out_1027.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/110000/225AD92E-83FD-3742-BCD3-5B19F4263977.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018B-UL2018_MiniAODv2-v2/out_1031.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/110000/719878CD-9869-F34E-A60D-03424553CB60.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018B-UL2018_MiniAODv2-v2/out_1021.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/110000/782CCD83-723E-BD4A-8409-6C91DDAD1610.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018B-UL2018_MiniAODv2-v2/out_1020.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/110000/D2C259D0-3109-3F4A-B7ED-EB39C16D2495.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018B-UL2018_MiniAODv2-v2/out_1023.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/260000/348C04CB-65ED-8643-A5CB-C462C56C5BD6.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018B-UL2018_MiniAODv2-v2/out_114.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/110000/6EE0FEAB-8DC5-1043-88E6-3AB8D5B0750A.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018B-UL2018_MiniAODv2-v2/out_1036.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/110000/06B38D78-9A5C-8940-9221-995996CA7301.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018B-UL2018_MiniAODv2-v2/out_1029.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/110000/F4796FF0-CFDC-2941-B015-BA8D338F6F53.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018B-UL2018_MiniAODv2-v2/out_1028.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/110000/A59236AB-F909-1943-AAF7-54DBE092BAC9.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018B-UL2018_MiniAODv2-v2/out_1037.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/B4445300-C8C5-2242-8B6A-F7B5297ABE66.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018B-UL2018_MiniAODv2-v2/out_726.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/728A0280-EF61-194E-99C5-E0CBC777BB32.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018B-UL2018_MiniAODv2-v2/out_727.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/79157CCF-4816-5344-B4AB-F7E98831F59E.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018B-UL2018_MiniAODv2-v2/out_724.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/260000/E87256BC-4DFE-8E49-8D2F-26A536786DB1.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018B-UL2018_MiniAODv2-v2/out_89.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/B815E925-6A46-C54C-B8AF-CC8FE88F0AA1.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018B-UL2018_MiniAODv2-v2/out_568.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/110000/2DB7EE0D-3D8A-D944-BD68-C068B6D1A8A0.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018B-UL2018_MiniAODv2-v2/out_1034.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/C5FB1209-172D-1F42-A05A-0DAD001A7FBB.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018B-UL2018_MiniAODv2-v2/out_720.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/8C996AFA-0705-1A49-AE1A-6566E95CC008.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018B-UL2018_MiniAODv2-v2/out_721.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/110000/82D96054-1442-F645-8EF1-6A8AD690E865.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018B-UL2018_MiniAODv2-v2/out_1035.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/110000/62412EA5-2C13-7948-91B1-74FBA7B6E4EC.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018B-UL2018_MiniAODv2-v2/out_728.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/110000/BE396DA3-2DE8-704F-95C3-879274832D30.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018B-UL2018_MiniAODv2-v2/out_729.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/93118090-2236-5149-AF93-7C9E04851BA6.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018B-UL2018_MiniAODv2-v2/out_605.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/FF1860AF-A5E1-F544-9BA9-90632C120597.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018B-UL2018_MiniAODv2-v2/out_604.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/74C7DCB5-929F-F34B-9BD4-3A87351D8FF4.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018B-UL2018_MiniAODv2-v2/out_607.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/E30E1EEA-8E27-A344-B1DE-6DEC0529F726.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018B-UL2018_MiniAODv2-v2/out_606.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/E2F9E819-D2BB-ED4C-81C2-E68F1DF55A1F.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018B-UL2018_MiniAODv2-v2/out_601.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/543C8DF5-FA89-484E-9AB9-44A6D6FD0E38.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018B-UL2018_MiniAODv2-v2/out_600.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/449EE00A-C7EA-AE44-89F0-810EE2AA1379.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018B-UL2018_MiniAODv2-v2/out_603.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/4FCF8BD1-4FD8-7E49-B6B9-7B5085B9584B.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018B-UL2018_MiniAODv2-v2/out_602.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/270000/8F1B97AF-086D-F446-BE7F-8A29154707F7.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018B-UL2018_MiniAODv2-v2/out_159.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/3BA3EBDE-AE8C-DF4F-B971-E1AFE8232A81.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018B-UL2018_MiniAODv2-v2/out_158.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/260000/64BDB8A9-A3D2-7B47-B07B-D72D0D9D9F2D.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018B-UL2018_MiniAODv2-v2/out_112.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/FA042BD9-D09C-B040-B64C-9D98C4E2824F.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018B-UL2018_MiniAODv2-v2/out_609.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/CD148B5C-96DD-754B-AF11-ED56660E3B54.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018B-UL2018_MiniAODv2-v2/out_608.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/92B5FA0A-5F10-B54A-9B18-87030E596E21.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018B-UL2018_MiniAODv2-v2/out_976.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/9BB7B7C7-574D-6B4E-A865-1403E71EC025.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018B-UL2018_MiniAODv2-v2/out_634.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/110000/FF6BDFFB-5647-FA4F-89E9-BB1EF9BED9CB.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018B-UL2018_MiniAODv2-v2/out_935.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/61AC0597-8F3D-A044-8CC8-2F313A2A58EA.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018B-UL2018_MiniAODv2-v2/out_399.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/4595BD57-5440-9949-807F-FE63DD7AE15B.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018B-UL2018_MiniAODv2-v2/out_635.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/717C2A87-2F33-D24C-BD8D-C0952C000D2D.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018B-UL2018_MiniAODv2-v2/out_959.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/92602630-9F13-814F-A031-60D3DAA55B24.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018B-UL2018_MiniAODv2-v2/out_958.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/4D4171A5-1327-F341-B6F0-13293F8BA6C6.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018B-UL2018_MiniAODv2-v2/out_398.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/260000/44BD0B44-FF69-234D-B3A3-E7D1A8E12BD0.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018B-UL2018_MiniAODv2-v2/out_48.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/260000/3E75F9D8-3186-5049-B993-0E2E67090F09.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018B-UL2018_MiniAODv2-v2/out_49.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/110000/1E263728-FF76-3244-B232-4C30D219C740.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018B-UL2018_MiniAODv2-v2/out_951.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/110000/89666024-08FB-5E47-891E-96DD0DB4F94A.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018B-UL2018_MiniAODv2-v2/out_950.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/110000/F58625D8-1DF1-7946-8ADC-A45C957F4786.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018B-UL2018_MiniAODv2-v2/out_953.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/110000/E9FB51F4-0DD2-3148-A474-C8CD5616F2B1.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018B-UL2018_MiniAODv2-v2/out_952.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/260000/91899892-BECC-8244-8BEB-5D4CA3BB57E3.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018B-UL2018_MiniAODv2-v2/out_42.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/110000/4ABE1750-B499-3846-9F19-18C68D33CE60.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018B-UL2018_MiniAODv2-v2/out_954.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/110000/9C33A915-0CFF-6345-81D0-B5FC9C2FB7BB.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018B-UL2018_MiniAODv2-v2/out_957.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/110000/E5706057-1FDB-7C49-BCBE-449BDE928BEF.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018B-UL2018_MiniAODv2-v2/out_956.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/E17DAA55-A826-6242-9692-81924B47DD8F.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018B-UL2018_MiniAODv2-v2/out_638.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/260000/E7B1A7FA-2A9A-C443-A09D-165E0DD8B03C.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018B-UL2018_MiniAODv2-v2/out_5.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/F5FA6D2D-3057-A742-B515-D5513C0D558A.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018B-UL2018_MiniAODv2-v2/out_639.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/41AC999E-CA38-3C4D-B845-828B6182F049.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018B-UL2018_MiniAODv2-v2/out_620.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/F92F78D6-D214-374F-B809-0AB0946B1B6B.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018B-UL2018_MiniAODv2-v2/out_460.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/96ED29D9-DAE6-EA43-B5E8-FA60D4FC73D0.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018B-UL2018_MiniAODv2-v2/out_489.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/3B569B2D-EF09-0948-8267-6A00ABF8B15A.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018B-UL2018_MiniAODv2-v2/out_488.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/B78515FF-EE06-2543-84B0-7B1D715DB71E.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018B-UL2018_MiniAODv2-v2/out_487.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/8D921476-A6CD-9A40-99EB-6BDAD076F7C4.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018B-UL2018_MiniAODv2-v2/out_486.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/6570B90D-B194-0B47-8201-DBD21763638A.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018B-UL2018_MiniAODv2-v2/out_869.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/720EF3A1-B82C-FE48-89EF-500B5FD4D4B1.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018B-UL2018_MiniAODv2-v2/out_484.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/FCD67DEE-0D30-4046-9160-037679A3E45B.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018B-UL2018_MiniAODv2-v2/out_483.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/14A2B7F7-BCF6-3D44-8CF4-091B6824649F.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018B-UL2018_MiniAODv2-v2/out_482.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/1E3E6F7F-AFC0-3B46-9CDD-9D14F3CC9455.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018B-UL2018_MiniAODv2-v2/out_481.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/0B16EFF9-6B4A-914E-AB8B-BCCD94620498.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018B-UL2018_MiniAODv2-v2/out_480.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/E2F5590B-B262-FB49-B6BA-265C6B63329E.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018B-UL2018_MiniAODv2-v2/out_509.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/110000/F641B5EC-AF08-F945-B67F-2CC4FDB56553.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018B-UL2018_MiniAODv2-v2/out_955.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/C9C44D58-9FE7-4F46-9479-6386E39934CC.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018B-UL2018_MiniAODv2-v2/out_472.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/31076AA8-1F96-6347-B53B-EF5C6ABC5FF8.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018B-UL2018_MiniAODv2-v2/out_473.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/1CEAD298-0892-4843-AABD-F16F71D30F70.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018B-UL2018_MiniAODv2-v2/out_470.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/DCBC358C-D82F-7340-BDD1-C32726C4AF9F.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018B-UL2018_MiniAODv2-v2/out_471.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/A7A785FC-5C92-1B40-ABBE-4682F8E3084C.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018B-UL2018_MiniAODv2-v2/out_476.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/E9BC37DC-297D-584F-85F7-298F0F846684.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018B-UL2018_MiniAODv2-v2/out_477.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/BD32F4A1-65F4-E045-A7FF-B73E80345BCB.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018B-UL2018_MiniAODv2-v2/out_474.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/455E976B-2778-3243-B394-4AAFC01A9F94.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018B-UL2018_MiniAODv2-v2/out_475.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/260000/2A48215A-8558-5A43-B30F-F8AD3831F04B.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018B-UL2018_MiniAODv2-v2/out_46.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/A47882B9-9614-1E4F-9B74-DFE3305C018A.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018B-UL2018_MiniAODv2-v2/out_479.root -a '-o ./ -d --year UL2018B'"
queue