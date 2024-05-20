universe=Vanilla
+ProjectName = "cms.org"
requirements = (OpSysAndVer =?= "CentOS7")
RequestMemory = 2048
RequestCpus = 1
executable=/home/legao/Wcb/condor/production/NanoNtupleChain_25_Apr_2024/JetHT_Run2018B-UL2018_MiniAODv2-v1/exe_CUSTNANO_UL18_MINIAODv2_DATA.sh
transfer_executable=True
transfer_input_files=/home/legao/Wcb/condor/x509up_u14433,/home/legao/Wcb/condor/Scripts/fetchFiles.py
transfer_output_files=""
log=/home/legao/Wcb/condor/production/NanoNtupleChain_25_Apr_2024/JetHT_Run2018B-UL2018_MiniAODv2-v1/log//$(Cluster).log
output=/home/legao/Wcb/condor/production/NanoNtupleChain_25_Apr_2024/JetHT_Run2018B-UL2018_MiniAODv2-v1/log//$(Cluster).out
error=/home/legao/Wcb/condor/production/NanoNtupleChain_25_Apr_2024/JetHT_Run2018B-UL2018_MiniAODv2-v1/log//$(Cluster).err
notification=Never
should_transfer_files = YES
when_to_transfer_output = ON_EXIT
x509userproxy=x509up_u14433
+MaxRuntime=150000
on_exit_remove        = (ExitBySignal == False) && (ExitCode == 0)
on_exit_hold          = (ExitBySignal == True) || (ExitCode != 0)
on_exit_hold_reason   = strcat("Job held by ON_EXIT_HOLD due to ", ifThenElse((ExitBySignal == True), strcat("exit signal ", ExitSignal), strcat("exit code ", ExitCode)), ".")
periodic_release      = (NumJobStarts < 3) && ((CurrentTime - EnteredCurrentStatus) > 60*60)
arguments="-f /store/data/Run2018B/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260001/24A067F4-9D0E-A44B-B5B4-934FCC1A947A.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018B-UL2018_MiniAODv2-v1/out_723.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260001/DCB92535-598A-8847-8A98-292078213D7E.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018B-UL2018_MiniAODv2-v1/out_198.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260001/9DAA66B9-9723-9848-804A-B55A4B5FD2A9.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018B-UL2018_MiniAODv2-v1/out_669.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260001/740DCD66-E1D9-6A47-9F1A-3B5F2F13EC79.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018B-UL2018_MiniAODv2-v1/out_668.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260000/CF72DF9B-A5FF-0945-89D0-8FF4868CEDE1.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018B-UL2018_MiniAODv2-v1/out_344.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260001/C7230DA2-FD6D-E942-899D-3887D1205573.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018B-UL2018_MiniAODv2-v1/out_819.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260000/1881CC1C-D76A-0F4A-B76D-AC4162FC2F79.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018B-UL2018_MiniAODv2-v1/out_346.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260000/270CC461-E5FA-DA4E-9C18-AAD9DAD13D3F.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018B-UL2018_MiniAODv2-v1/out_347.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260001/4716E255-D413-8541-97C9-40AA3BF4FF8E.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018B-UL2018_MiniAODv2-v1/out_340.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260001/CF9501DB-F07E-5C4F-B514-E0F3A7823B47.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018B-UL2018_MiniAODv2-v1/out_341.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260000/BD9AEDA7-51B0-524F-AE8E-CA9C1DEF87D3.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018B-UL2018_MiniAODv2-v1/out_342.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260000/9CD5A324-2005-C44B-BAB9-C5327B74603E.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018B-UL2018_MiniAODv2-v1/out_343.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260000/6A5FBF70-72E4-2E4F-B176-F8C1AD588D25.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018B-UL2018_MiniAODv2-v1/out_810.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260000/7ED6F5EF-B414-054E-9807-A7248C595607.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018B-UL2018_MiniAODv2-v1/out_811.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260000/3DE911FB-8C4D-974E-9E65-BD458CD4CEB6.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018B-UL2018_MiniAODv2-v1/out_812.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260000/FF820996-2A9F-354E-8D50-E95FAAADD7F7.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018B-UL2018_MiniAODv2-v1/out_813.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260000/3EBAC641-24FA-4E4D-A692-3F73F7D1A413.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018B-UL2018_MiniAODv2-v1/out_348.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260000/23C246EB-1602-5D4D-9547-4B914195FE31.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018B-UL2018_MiniAODv2-v1/out_349.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260000/22BDA98D-10E8-1C49-9024-5A4948D69E55.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018B-UL2018_MiniAODv2-v1/out_816.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260000/5C6C9B17-87A3-FB49-9909-64D12E1EEA1C.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018B-UL2018_MiniAODv2-v1/out_817.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260000/7DC7606E-EAE4-C942-9143-211B7FCED45E.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018B-UL2018_MiniAODv2-v1/out_664.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260001/67444D91-37C8-864A-AEF9-F14F1DA074B8.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018B-UL2018_MiniAODv2-v1/out_739.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260001/206082A9-CA70-C641-B601-924F23E9142B.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018B-UL2018_MiniAODv2-v1/out_719.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260000/E3B04F5E-DE1D-C34C-8C20-9CCA3574F6BB.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018B-UL2018_MiniAODv2-v1/out_718.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260000/D2F34F50-CCEB-2141-BF23-F74CD3963C91.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018B-UL2018_MiniAODv2-v1/out_717.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260001/8B750C2C-395A-394C-AA31-BB7653CBE042.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018B-UL2018_MiniAODv2-v1/out_716.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260001/59727192-7EEB-E147-A311-23AEE348DFAE.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018B-UL2018_MiniAODv2-v1/out_715.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260001/7386DEC3-995B-034A-AA82-82F2045F7DA6.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018B-UL2018_MiniAODv2-v1/out_714.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260001/566E8C03-4C8E-0540-B070-9477BBA79A6C.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018B-UL2018_MiniAODv2-v1/out_713.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260001/0E78CFEA-5969-D647-AC47-910B89F6EB72.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018B-UL2018_MiniAODv2-v1/out_712.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260000/6697971B-4083-0E48-B191-58C0E7D0B945.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018B-UL2018_MiniAODv2-v1/out_711.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260000/0BE63E91-D024-F54C-904F-143C6D9DB6AE.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018B-UL2018_MiniAODv2-v1/out_710.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260001/B7AA91EC-7380-ED4F-B22C-E4C1A2AA8A33.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018B-UL2018_MiniAODv2-v1/out_618.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260000/9937DFE9-B47E-1A48-84B7-73916CD89FB0.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018B-UL2018_MiniAODv2-v1/out_421.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260000/3BFA021B-6486-3349-9AEB-7AB2186E5C6B.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018B-UL2018_MiniAODv2-v1/out_129.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260000/8D468C93-F9E2-8E45-9596-643AEBCDDF6D.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018B-UL2018_MiniAODv2-v1/out_420.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260000/82B5675E-8579-414A-83D8-604D155FA11F.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018B-UL2018_MiniAODv2-v1/out_423.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260002/89713699-E1E7-0D42-A270-00461AD0C1C7.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018B-UL2018_MiniAODv2-v1/out_1069.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260002/52287D92-A551-AD4B-88A2-A618F20FAB6B.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018B-UL2018_MiniAODv2-v1/out_1068.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260000/7EC06EFA-3FD1-704B-B03A-356560794A41.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018B-UL2018_MiniAODv2-v1/out_422.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260002/67042F89-6298-AD46-B9C1-3DC5000ED4F6.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018B-UL2018_MiniAODv2-v1/out_1061.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260002/D47C5755-67C5-1A42-A945-43D88024ABDF.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018B-UL2018_MiniAODv2-v1/out_1060.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260002/A3D88958-4EFA-A24A-9820-12ABEE4CDE29.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018B-UL2018_MiniAODv2-v1/out_1063.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260002/D3D1F7D3-8798-B64C-93BC-BC2EE097F45B.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018B-UL2018_MiniAODv2-v1/out_1062.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260002/DAED19C8-7B70-CF4D-B180-6F0C79B66A02.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018B-UL2018_MiniAODv2-v1/out_1065.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260002/3828E24C-9761-E448-AD75-BF92BFD73192.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018B-UL2018_MiniAODv2-v1/out_1064.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260002/0A11AB82-B3EA-9A48-A1B6-5CCB84627002.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018B-UL2018_MiniAODv2-v1/out_1067.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260002/949514C3-B0DF-8742-A44E-40B3704F1776.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018B-UL2018_MiniAODv2-v1/out_1066.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260001/96CAED9D-259B-354B-BA15-DA0F260F7438.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018B-UL2018_MiniAODv2-v1/out_671.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260000/E42B36C6-32D2-1742-98C7-8B9BD70A89C6.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018B-UL2018_MiniAODv2-v1/out_427.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260000/26CD02D2-7381-504E-8DE2-4176433D0CAF.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018B-UL2018_MiniAODv2-v1/out_298.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260000/BA90F0E4-698C-7B48-98C1-07369C61B9D0.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018B-UL2018_MiniAODv2-v1/out_299.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260000/A5F1C19D-0CD7-9F45-BC16-109564B3D139.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018B-UL2018_MiniAODv2-v1/out_296.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260000/913D194C-C22D-0A4D-B392-EDB41FB4864A.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018B-UL2018_MiniAODv2-v1/out_297.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260000/6198FDEB-9883-C14E-AEE4-7E4360B8374D.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018B-UL2018_MiniAODv2-v1/out_294.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260000/829F1DBF-F216-3F4E-9287-E501CD2786F0.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018B-UL2018_MiniAODv2-v1/out_295.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260000/75BA9216-F457-794C-B26F-5B324567478E.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018B-UL2018_MiniAODv2-v1/out_292.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260000/F8FCFAFF-05C2-584B-B419-1C3D6F849F35.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018B-UL2018_MiniAODv2-v1/out_293.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260000/4AE27C34-E360-724C-BC65-C09400376BA0.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018B-UL2018_MiniAODv2-v1/out_290.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260000/0873741D-7529-2B46-8485-9B5FD10A33A1.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018B-UL2018_MiniAODv2-v1/out_291.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260000/4E3539EA-7878-B54A-9308-C896FE85A01A.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018B-UL2018_MiniAODv2-v1/out_591.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260000/0E6C48C5-AFF8-1340-9825-8D70A4443835.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018B-UL2018_MiniAODv2-v1/out_590.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260000/5A88DA43-EA90-CA45-8137-C2E5B268DC71.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018B-UL2018_MiniAODv2-v1/out_593.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260000/AB3FE756-11AD-954C-B913-A98E7605FE29.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018B-UL2018_MiniAODv2-v1/out_592.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260001/641952C5-225E-B643-B77C-3AE11A0E55AC.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018B-UL2018_MiniAODv2-v1/out_595.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260002/CF71B36C-4BA6-2340-9651-9EBCE9D0F0EE.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018B-UL2018_MiniAODv2-v1/out_1082.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260001/44B52748-BB51-4A42-911C-6CA4333ACF6C.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018B-UL2018_MiniAODv2-v1/out_597.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260001/28816376-B907-3648-9578-D53F656FC852.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018B-UL2018_MiniAODv2-v1/out_596.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260001/BDA5B710-0AA9-6C45-864E-A93F6116207F.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018B-UL2018_MiniAODv2-v1/out_195.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260001/7A5A1481-6DFC-5743-A897-4B0E18AE44B0.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018B-UL2018_MiniAODv2-v1/out_194.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260001/BBB2AD23-DEBD-534B-961A-1B022D2505D7.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018B-UL2018_MiniAODv2-v1/out_197.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260001/90F01971-99F8-5C49-A46D-DE6AF37F7D31.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018B-UL2018_MiniAODv2-v1/out_196.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260001/1853167A-BBE0-824A-8D9C-7339F760485C.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018B-UL2018_MiniAODv2-v1/out_191.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260001/72AC4AF6-6915-1444-89BE-2071AC82018C.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018B-UL2018_MiniAODv2-v1/out_190.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260001/F401F509-09B8-6B45-AE23-F25383355899.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018B-UL2018_MiniAODv2-v1/out_193.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260001/36889231-671A-EC4C-B74B-D064E2AD42BA.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018B-UL2018_MiniAODv2-v1/out_192.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260000/4AEFB92A-39FB-CE48-91EC-F9DCA4992DBF.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018B-UL2018_MiniAODv2-v1/out_270.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260000/8FEC2EC5-3D87-2946-BEEB-2FDA2DD380D8.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018B-UL2018_MiniAODv2-v1/out_271.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260000/D0B6FF90-8129-3E43-A876-85997D3716BE.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018B-UL2018_MiniAODv2-v1/out_272.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260000/F2EC4BEA-73EB-C042-9D13-08BBBAEE3F1E.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018B-UL2018_MiniAODv2-v1/out_273.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260000/2A5AC206-BC20-7041-A11A-00CBDC2B1EB1.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018B-UL2018_MiniAODv2-v1/out_274.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260000/87364CC4-FA94-6F45-B61D-3C0A4CFA4C28.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018B-UL2018_MiniAODv2-v1/out_275.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260000/B751D3FB-70F3-8E42-88D7-A9D4D6F31D5E.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018B-UL2018_MiniAODv2-v1/out_276.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260000/138D0003-22C5-FE4A-AB7E-2D97A6E3E996.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018B-UL2018_MiniAODv2-v1/out_277.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260000/3CD85340-530D-3346-97F2-D7CE9668A449.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018B-UL2018_MiniAODv2-v1/out_278.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260000/7D48F2B5-1C0B-9447-8665-B5CE13459BDC.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018B-UL2018_MiniAODv2-v1/out_279.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260000/5067788F-FA2A-764C-AE3C-BABD4628CD88.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018B-UL2018_MiniAODv2-v1/out_569.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260000/B8E04D97-3932-C347-8A51-7EEA6760ABC6.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018B-UL2018_MiniAODv2-v1/out_497.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260001/017AACE0-B43D-6642-9A0D-42DD3BDB6F28.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018B-UL2018_MiniAODv2-v1/out_524.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260001/9E99432D-9B1E-E24D-9A14-B3C08921A5AE.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018B-UL2018_MiniAODv2-v1/out_525.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260001/E1A2BEF1-C8A6-D84D-AF55-E5D3EE5FEBA1.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018B-UL2018_MiniAODv2-v1/out_526.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260001/4731012C-97BB-724D-BDD0-B96BD5909FBB.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018B-UL2018_MiniAODv2-v1/out_527.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260000/93743C19-EBAB-AD4F-8B83-1EF20D9B7AAC.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018B-UL2018_MiniAODv2-v1/out_520.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260000/3D6AA4A4-702E-B94B-A7A2-474D2A71C3C2.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018B-UL2018_MiniAODv2-v1/out_521.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260000/39F69C5B-7D45-8D44-8402-4A81F54333FE.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018B-UL2018_MiniAODv2-v1/out_522.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260000/A35E8A66-7C67-3F44-9F71-847ECAD03F8D.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018B-UL2018_MiniAODv2-v1/out_523.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260002/739D6049-7B8A-5C4E-8E88-49B92210C656.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018B-UL2018_MiniAODv2-v1/out_1014.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260002/8A4F0301-58CC-784A-8AAD-62F4DA2B5A1B.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018B-UL2018_MiniAODv2-v1/out_1015.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260001/1BFB0C9D-0D9E-8342-89F4-ECA2B479DB1B.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018B-UL2018_MiniAODv2-v1/out_599.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260002/3E2685AA-DD67-C147-9273-6C41DD8F8ECF.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018B-UL2018_MiniAODv2-v1/out_1017.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260001/1529A5B0-5823-FB4F-83C8-814EC2C6315D.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018B-UL2018_MiniAODv2-v1/out_528.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260001/F8A5CDF1-DA3D-5F49-9299-7BFAB92AD1EA.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018B-UL2018_MiniAODv2-v1/out_529.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260002/616F6C59-FC4A-934C-8958-BB16C9BBC29C.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018B-UL2018_MiniAODv2-v1/out_1012.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260000/A00525E7-BB8B-5C47-B06A-89D52557ECCD.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018B-UL2018_MiniAODv2-v1/out_1013.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/JetHT/MINIAOD/UL2018_MiniAODv2-v1/50000/B60734C6-52D6-FA45-B673-712AC5041643.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018B-UL2018_MiniAODv2-v1/out_1234.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/JetHT/MINIAOD/UL2018_MiniAODv2-v1/50000/FFBB0084-7AF3-CE4A-AF41-96F6E2A5D14E.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018B-UL2018_MiniAODv2-v1/out_1235.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260000/56EC962E-F255-0441-A672-F3BF242C1DB2.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018B-UL2018_MiniAODv2-v1/out_449.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260000/D6ABF766-2240-5845-905D-F23E28C21238.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018B-UL2018_MiniAODv2-v1/out_448.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260000/13717BBD-585F-424A-AFDE-969FEDE37997.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018B-UL2018_MiniAODv2-v1/out_1230.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/JetHT/MINIAOD/UL2018_MiniAODv2-v1/50000/1B6F2FF9-620D-A545-BEAC-DB26BAF557A6.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018B-UL2018_MiniAODv2-v1/out_1231.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/JetHT/MINIAOD/UL2018_MiniAODv2-v1/50000/DCF468F8-785B-F24B-84B4-9100452F8E2C.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018B-UL2018_MiniAODv2-v1/out_1232.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/JetHT/MINIAOD/UL2018_MiniAODv2-v1/50000/B69A7A13-B34C-AB4D-A33D-E24B98C9C72D.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018B-UL2018_MiniAODv2-v1/out_1233.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260000/8E13CABC-F150-9D4D-A9A9-57C97B54BD51.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018B-UL2018_MiniAODv2-v1/out_443.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260000/E63419DB-0212-EE47-9E9B-938A93D3BE87.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018B-UL2018_MiniAODv2-v1/out_442.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260000/38197D77-EAEC-CE42-87AB-4F738EE51C6B.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018B-UL2018_MiniAODv2-v1/out_441.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260000/A0C8918F-7ABF-F940-95BD-427EB193A50C.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018B-UL2018_MiniAODv2-v1/out_440.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260000/1B0FC9F6-7832-0A44-836A-47BA1BE04D5F.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018B-UL2018_MiniAODv2-v1/out_447.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260000/03D5CFC7-63E1-934E-9EB8-ED478870BB5B.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018B-UL2018_MiniAODv2-v1/out_446.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260000/55FB4C77-403E-B541-B88C-C1243427BE1A.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018B-UL2018_MiniAODv2-v1/out_445.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260000/5F66CBAD-A240-6E44-8FCD-5C4517F15767.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018B-UL2018_MiniAODv2-v1/out_444.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260001/03B2352D-4B12-3F4F-B382-ABFBB2D41BBB.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018B-UL2018_MiniAODv2-v1/out_47.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260001/2FDCF425-329F-9948-A0C6-F922D057792B.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018B-UL2018_MiniAODv2-v1/out_108.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260001/4D509C87-361A-4B40-A334-D3F233200FE6.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018B-UL2018_MiniAODv2-v1/out_109.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260000/CFA39237-6F9A-8D4F-A65C-74EDD0DA4ABF.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018B-UL2018_MiniAODv2-v1/out_588.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260001/001FDC41-E2B1-7C42-84CD-F48A143AA3F2.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018B-UL2018_MiniAODv2-v1/out_102.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260001/FB34D9D1-97C8-5342-9667-3EAAAD3176CB.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018B-UL2018_MiniAODv2-v1/out_103.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260001/C763CE95-694A-3140-A899-B58B01B9F38E.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018B-UL2018_MiniAODv2-v1/out_100.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260001/628D9666-8ACD-1243-A4DA-AEE577A92234.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018B-UL2018_MiniAODv2-v1/out_101.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260001/8BFE11FE-3345-E040-B530-05EBF3342CCF.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018B-UL2018_MiniAODv2-v1/out_106.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260001/C7A9BD82-98F9-5949-9D4A-66237D314867.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018B-UL2018_MiniAODv2-v1/out_107.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260001/6FEB14AE-6194-1745-845F-2B482014DF59.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018B-UL2018_MiniAODv2-v1/out_104.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260001/71E60E7D-A659-504F-8EFE-C6A3C10522C4.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018B-UL2018_MiniAODv2-v1/out_105.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260000/123D79DC-F1DF-AE49-AB3B-54B0C4EF908F.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018B-UL2018_MiniAODv2-v1/out_902.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260000/1E4B9B53-421B-3341-9718-5CB2AE13C4AA.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018B-UL2018_MiniAODv2-v1/out_903.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260000/B8C653D2-C9B2-3E44-A637-B6C0DC7197F2.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018B-UL2018_MiniAODv2-v1/out_39.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260002/28F18341-4031-134A-B670-B3ED30CE96BE.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018B-UL2018_MiniAODv2-v1/out_38.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260000/C57843F2-4C0A-1946-A7B9-664E68E46D63.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018B-UL2018_MiniAODv2-v1/out_906.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260000/B17D905F-4B9F-3248-9B5F-534FC89CF7A4.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018B-UL2018_MiniAODv2-v1/out_907.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260000/CE814359-137F-C248-AD12-231709D08A8B.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018B-UL2018_MiniAODv2-v1/out_904.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260000/D5A55A2C-4A00-2849-8A3F-085A7587312B.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018B-UL2018_MiniAODv2-v1/out_905.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260001/DBEDCBDF-4068-B34E-B119-585A020340CA.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018B-UL2018_MiniAODv2-v1/out_33.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260001/81758C62-9302-C042-90E1-C54D12E52689.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018B-UL2018_MiniAODv2-v1/out_32.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260001/B095E0D5-89FD-6C44-BB9B-A5B34FC8CB48.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018B-UL2018_MiniAODv2-v1/out_31.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260001/7EBD1C18-8557-0344-AE55-FAC43AB21ACC.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018B-UL2018_MiniAODv2-v1/out_30.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260002/42A00421-AE12-FD4A-8F05-B6715628781D.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018B-UL2018_MiniAODv2-v1/out_37.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260001/BFEC57D1-B6F4-E84A-B963-53228B6B523C.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018B-UL2018_MiniAODv2-v1/out_36.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260001/8C510560-CEDC-3241-B572-5374D4F2B681.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018B-UL2018_MiniAODv2-v1/out_35.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260001/2F717B63-0765-EB4F-B8A7-97023B53AFCD.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018B-UL2018_MiniAODv2-v1/out_34.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260000/28924AD9-FFB6-2247-85EE-A91778503E68.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018B-UL2018_MiniAODv2-v1/out_641.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260000/E5DDBA27-9013-AB44-A198-5CD301A38F7B.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018B-UL2018_MiniAODv2-v1/out_640.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260000/4EFD1749-F580-2C44-9C7A-FC5676E37B5C.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018B-UL2018_MiniAODv2-v1/out_643.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260000/55BF89B9-3AA5-3145-8F02-FF7810985768.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018B-UL2018_MiniAODv2-v1/out_642.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260000/DAC32487-DF0E-EC4B-BA76-3E36C5FF1F9F.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018B-UL2018_MiniAODv2-v1/out_645.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260000/40F4D9B1-8CDB-DA4D-BEE4-4FA37A44DA8D.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018B-UL2018_MiniAODv2-v1/out_644.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260000/6DF47CDD-B923-2241-B1E3-5F956E0DE17F.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018B-UL2018_MiniAODv2-v1/out_438.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260000/25875F82-686D-4547-8A45-1F34CBBDA264.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018B-UL2018_MiniAODv2-v1/out_439.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260000/03BBA139-5B67-8444-96C9-54675897AC2F.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018B-UL2018_MiniAODv2-v1/out_436.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260000/58DABDAE-AE13-2D42-BEB7-85A0A478D88F.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018B-UL2018_MiniAODv2-v1/out_437.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260000/FEFAAD71-846C-BF4E-BA93-108F74185E94.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018B-UL2018_MiniAODv2-v1/out_434.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260000/3EBFE728-9C19-E344-96BA-B6C8C3F71ABA.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018B-UL2018_MiniAODv2-v1/out_435.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260000/EEE3E810-2858-0D4B-8CF2-C00DD2864E24.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018B-UL2018_MiniAODv2-v1/out_432.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260000/9446FE14-0D7D-EF47-960F-C523B8AAAFEE.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018B-UL2018_MiniAODv2-v1/out_433.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260000/0AF8F532-36DF-E44E-8232-EAF6449FA113.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018B-UL2018_MiniAODv2-v1/out_430.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260000/3657AE53-FCC8-CE4A-B060-894C2BC5D38D.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018B-UL2018_MiniAODv2-v1/out_431.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260000/DFFB13A9-7174-2B49-A1A5-DB1984D12E84.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018B-UL2018_MiniAODv2-v1/out_826.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260001/C3FA7EBC-25EF-0A48-B5B5-D82CE19E9B2F.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018B-UL2018_MiniAODv2-v1/out_339.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260001/CA0F8C1B-D48F-3A46-AAC5-9AEB1DCDE75E.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018B-UL2018_MiniAODv2-v1/out_338.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260001/2B86F013-C247-6946-B278-BB0DAD88CD56.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018B-UL2018_MiniAODv2-v1/out_335.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260001/9276D69F-C210-CA4C-B1E4-F4F43B71D227.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018B-UL2018_MiniAODv2-v1/out_334.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260001/BEB90C52-5933-724A-B05B-DEE6FD6E55CE.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018B-UL2018_MiniAODv2-v1/out_337.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260001/713581BC-65EA-5C49-A223-626A13220B6A.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018B-UL2018_MiniAODv2-v1/out_336.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260001/3397C293-2278-CC41-B0CF-F430C0FA0EE0.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018B-UL2018_MiniAODv2-v1/out_331.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260001/76831EA3-5103-4345-9985-CEF8B3644457.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018B-UL2018_MiniAODv2-v1/out_330.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260001/04778F8E-7372-4C4B-B345-F52AA820C056.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018B-UL2018_MiniAODv2-v1/out_333.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260001/4377FA61-3C53-7444-B042-A3252362744F.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018B-UL2018_MiniAODv2-v1/out_332.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260000/BB26A5CD-3A82-B443-A083-8E947628FB45.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018B-UL2018_MiniAODv2-v1/out_559.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260002/A64DBB04-D8C0-8845-847C-2FCA62EA9FCD.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018B-UL2018_MiniAODv2-v1/out_1026.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260001/81BE586F-9A38-C347-8583-E094D218CFA8.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018B-UL2018_MiniAODv2-v1/out_745.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260000/5D73CA8F-B556-1440-B273-EF173D52DCCE.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018B-UL2018_MiniAODv2-v1/out_854.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260001/9879D8AD-386B-2741-B78E-B7C7CFDF3EF1.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018B-UL2018_MiniAODv2-v1/out_818.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260001/D0AC0C0D-3654-A34A-A1C9-BB8EF3D34EE7.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018B-UL2018_MiniAODv2-v1/out_856.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260001/C9B7886E-9DF8-DD43-AB0B-665A799F1314.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018B-UL2018_MiniAODv2-v1/out_857.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260000/7374D63C-ACB8-4D41-BCBD-4EB084356B38.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018B-UL2018_MiniAODv2-v1/out_850.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260000/EBB89B2E-238F-584C-A7A8-3641EAD53624.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018B-UL2018_MiniAODv2-v1/out_851.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260000/376D5328-7A98-9E49-8940-47506DD1EAEF.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018B-UL2018_MiniAODv2-v1/out_852.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260001/DD240E78-B2B2-FA45-9542-C40E4A75A39F.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018B-UL2018_MiniAODv2-v1/out_0.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260001/76CE7D32-48A0-084D-BF01-51B3EBAA4A65.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018B-UL2018_MiniAODv2-v1/out_858.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260001/E4D025C9-AE4E-464D-AFA1-88B4F3D1E37C.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018B-UL2018_MiniAODv2-v1/out_859.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260000/8332F61A-89BC-B748-82CC-8939682BA435.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018B-UL2018_MiniAODv2-v1/out_748.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260001/54B836FF-8CAB-1642-8011-439CFF4068D8.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018B-UL2018_MiniAODv2-v1/out_6.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260000/F7BA5380-73F9-3E4C-B82B-050B37879463.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018B-UL2018_MiniAODv2-v1/out_151.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260000/ED3FC709-84B6-8E4C-86CE-8EA5D75099F9.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018B-UL2018_MiniAODv2-v1/out_425.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260000/A93D67D3-1E66-204B-BD10-C03BFBCB3D4D.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018B-UL2018_MiniAODv2-v1/out_634.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260000/D26A438E-A082-6D49-BED2-033148BCE375.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018B-UL2018_MiniAODv2-v1/out_43.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260001/A90491FA-CAAC-1440-988B-180A371BC8E4.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018B-UL2018_MiniAODv2-v1/out_99.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260001/738714C3-4CB3-7B41-AE40-84EF724D026E.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018B-UL2018_MiniAODv2-v1/out_98.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260001/59D6FA8C-D0C2-7A4F-B6B5-55AADFAC54E8.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018B-UL2018_MiniAODv2-v1/out_8.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260000/D201DC7B-3F55-EA45-8041-D89D04E7BC80.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018B-UL2018_MiniAODv2-v1/out_91.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260000/B8C461B8-F964-C04C-A857-FB5C124D6E72.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018B-UL2018_MiniAODv2-v1/out_90.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260000/BFA6A1C7-E869-B444-BC04-B907DF4DC7E9.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018B-UL2018_MiniAODv2-v1/out_93.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260000/D0B0CC4A-0BB4-4C44-99A0-01DF793BC687.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018B-UL2018_MiniAODv2-v1/out_92.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260001/202BF413-0DA3-3842-9AB6-3D19369434D3.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018B-UL2018_MiniAODv2-v1/out_95.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260000/76458128-3576-4B4C-8E34-B2F6C2C94B53.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018B-UL2018_MiniAODv2-v1/out_94.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260001/17C7F5A8-84FA-EE47-B291-1E247A5E5C46.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018B-UL2018_MiniAODv2-v1/out_97.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260001/30A27622-F372-804E-81BF-5CB72EA6E177.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018B-UL2018_MiniAODv2-v1/out_96.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260000/A2212423-DAFC-0B4B-81D1-D47363CBC333.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018B-UL2018_MiniAODv2-v1/out_150.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260000/1163998B-8E1D-2742-8B4A-98724B626C26.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018B-UL2018_MiniAODv2-v1/out_424.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260000/CE71635E-C8DB-124D-A262-4F27E0D8A71A.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018B-UL2018_MiniAODv2-v1/out_814.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260001/22982F38-AF18-0444-86FF-BA171C08A733.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018B-UL2018_MiniAODv2-v1/out_314.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260000/A85230CD-06F3-6B4B-8250-E8690E2B39F2.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018B-UL2018_MiniAODv2-v1/out_815.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260000/A18956AC-5CB6-0A46-B10B-03D61CAD64F4.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018B-UL2018_MiniAODv2-v1/out_153.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260001/2F8924F2-682A-564E-80B6-47E71F0C7475.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018B-UL2018_MiniAODv2-v1/out_740.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260001/8C26343E-6AD9-2347-BCCB-9514B5EA21C0.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018B-UL2018_MiniAODv2-v1/out_741.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260001/DACDB92E-0989-3041-8BF7-44CDFB5B6443.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018B-UL2018_MiniAODv2-v1/out_742.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260001/A20AE4EA-FB03-0D4D-8A1E-056F2C7FB52B.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018B-UL2018_MiniAODv2-v1/out_743.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260001/B7AEE257-C4CA-054E-A092-A41D9621A199.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018B-UL2018_MiniAODv2-v1/out_744.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260001/01B7A327-CEA1-6241-B8F4-2BE7BFA6D9C7.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018B-UL2018_MiniAODv2-v1/out_558.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260000/8B3D5533-328B-934A-B323-F85657E7DD43.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018B-UL2018_MiniAODv2-v1/out_746.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260000/0A2843DF-0672-4743-8653-35124090611E.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018B-UL2018_MiniAODv2-v1/out_747.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260001/B50E188B-A393-6449-B530-74E0915E0B70.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018B-UL2018_MiniAODv2-v1/out_555.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260001/49BD4C67-DC06-EE4F-82CA-5AA1DD7E4E29.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018B-UL2018_MiniAODv2-v1/out_554.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260001/22A7C90B-2F64-C249-B7E0-B6A5692C1516.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018B-UL2018_MiniAODv2-v1/out_557.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260001/9B778C60-4358-2C40-A359-9D5E3A1FDA56.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018B-UL2018_MiniAODv2-v1/out_556.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260001/821800D8-F2FA-4A47-A102-213E328302AA.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018B-UL2018_MiniAODv2-v1/out_551.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260001/B3FEDCDA-BE86-6645-BC02-89FC5E1FD9DC.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018B-UL2018_MiniAODv2-v1/out_550.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260001/DD41182C-8DC1-BE4A-B630-832074B3DCA6.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018B-UL2018_MiniAODv2-v1/out_553.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260001/3625B0DC-04DB-834C-8890-394934795E1E.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018B-UL2018_MiniAODv2-v1/out_552.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260000/DC5D8D4A-B704-FD4B-A7CD-B469F13722B7.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018B-UL2018_MiniAODv2-v1/out_238.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260000/AD7A8165-F9A8-A341-A3AC-FB27AFE734AC.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018B-UL2018_MiniAODv2-v1/out_239.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260000/A7D95D3E-7D89-8942-B1AF-3AB1809EBCCE.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018B-UL2018_MiniAODv2-v1/out_234.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260000/BD890DA6-79CC-D940-BFA1-616BC5B42044.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018B-UL2018_MiniAODv2-v1/out_235.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260000/C4802F75-363E-DF4D-B835-D16A1DFCCEA0.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018B-UL2018_MiniAODv2-v1/out_236.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260000/BF272722-F254-334B-8E3D-9C4CF3968E44.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018B-UL2018_MiniAODv2-v1/out_237.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260000/7900F811-23E8-9849-8470-707DD78B7465.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018B-UL2018_MiniAODv2-v1/out_230.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260000/BF4D5CB6-A2CB-FE4F-B370-3D15E67177B8.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018B-UL2018_MiniAODv2-v1/out_231.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260000/B19167E0-417F-E740-A818-9F9C9F80626B.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018B-UL2018_MiniAODv2-v1/out_232.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260000/5CE1FCA7-537E-DA41-BA0D-397CF2DE4C55.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018B-UL2018_MiniAODv2-v1/out_233.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260002/CBC0116D-F023-CA48-9237-6279FDED48BF.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018B-UL2018_MiniAODv2-v1/out_1050.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260002/3033DB62-FB80-A845-974D-61EC9097E34A.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018B-UL2018_MiniAODv2-v1/out_1051.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260000/3A5676F8-B421-BE4E-A32A-3B1751E73489.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018B-UL2018_MiniAODv2-v1/out_1052.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260002/E9D7CB8E-0005-3443-9303-764FB5844A17.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018B-UL2018_MiniAODv2-v1/out_1053.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260002/6E2E1A28-E174-014D-92D5-AA7CB599CD15.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018B-UL2018_MiniAODv2-v1/out_1054.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260002/E0BAB03B-F465-7C4D-AA73-4FDAE724523D.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018B-UL2018_MiniAODv2-v1/out_1055.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260002/153A2551-A591-594A-8711-C380831D74A3.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018B-UL2018_MiniAODv2-v1/out_1056.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260002/0C6BE380-8B1F-E845-870A-35E73420A076.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018B-UL2018_MiniAODv2-v1/out_1057.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260002/E05C8BF3-2E8D-974C-9BC0-07E169D20F4B.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018B-UL2018_MiniAODv2-v1/out_1058.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260002/A3215268-F48F-8247-85AD-45287F9A133F.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018B-UL2018_MiniAODv2-v1/out_1059.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260002/464B82F3-AD80-084E-8127-7219012B26D4.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018B-UL2018_MiniAODv2-v1/out_1193.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260002/87005C7E-5339-0542-82AB-3238B06C2FD0.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018B-UL2018_MiniAODv2-v1/out_1192.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260002/45264B30-4406-3047-92DA-AF45A117C9BD.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018B-UL2018_MiniAODv2-v1/out_1195.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260000/69FF70A9-4A3D-A84F-A933-59BCABCA827E.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018B-UL2018_MiniAODv2-v1/out_1194.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260002/70EADC01-4038-EA4D-ADF7-79E97B11E87D.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018B-UL2018_MiniAODv2-v1/out_1197.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260002/52D79815-01A6-9245-95BB-EF2038EADB0E.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018B-UL2018_MiniAODv2-v1/out_1196.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260001/AE24D678-8A4B-434B-BE3C-FD62BE34E601.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018B-UL2018_MiniAODv2-v1/out_1.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260002/397B2743-A640-C845-B495-1B5E9F4F941E.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018B-UL2018_MiniAODv2-v1/out_1159.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260000/B768788A-6BDD-B846-9C73-623E8784AB2F.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018B-UL2018_MiniAODv2-v1/out_155.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260000/9CAD8C75-2671-9041-8393-BD6C2A9FDB6E.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018B-UL2018_MiniAODv2-v1/out_159.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260000/4F4595B6-C94A-C44E-9CE4-398284690663.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018B-UL2018_MiniAODv2-v1/out_146.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260000/744BAE81-6C54-C14B-9BAE-688167822C4B.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018B-UL2018_MiniAODv2-v1/out_147.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260000/D19D87BF-AAB4-7A41-AD9C-51B8526F5C9E.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018B-UL2018_MiniAODv2-v1/out_144.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260000/4997A651-765A-E944-ADA6-907AFBC8B273.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018B-UL2018_MiniAODv2-v1/out_145.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260000/7EE10F7A-289E-F148-B371-6321DBB9DE38.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018B-UL2018_MiniAODv2-v1/out_142.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260000/07BEB0CF-3FFF-3C4B-99AF-177ECA1006B1.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018B-UL2018_MiniAODv2-v1/out_143.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260000/3A422D8D-A4EF-7140-8FE7-7F8926A84D36.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018B-UL2018_MiniAODv2-v1/out_140.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260000/F29577DE-D150-8B40-B27C-712F29DD4914.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018B-UL2018_MiniAODv2-v1/out_141.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260000/6A107F42-3CD6-CE48-B5A0-936853111114.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018B-UL2018_MiniAODv2-v1/out_612.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260000/3EAF0EE7-01F6-164E-B7EA-3863484F417C.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018B-UL2018_MiniAODv2-v1/out_613.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260001/D54258D0-0C3F-484D-8BE4-68DA766BCC88.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018B-UL2018_MiniAODv2-v1/out_610.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260001/B83D8F56-15C6-C24A-9307-A085A43A7904.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018B-UL2018_MiniAODv2-v1/out_611.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260000/C1C11A38-0E8C-E643-8F49-55F51C1C5775.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018B-UL2018_MiniAODv2-v1/out_616.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260001/C8F96D2A-D043-6644-9C05-3DE4CD3A4AE4.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018B-UL2018_MiniAODv2-v1/out_617.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260000/EFB48C5D-A288-5849-8891-BAA6A488BD4C.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018B-UL2018_MiniAODv2-v1/out_148.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260000/7970C724-3103-0145-830D-4345F0B35885.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018B-UL2018_MiniAODv2-v1/out_149.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260000/8724C8AA-FA26-7643-A1F1-9E88B6188840.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018B-UL2018_MiniAODv2-v1/out_511.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260000/B0529A70-9BB0-B24A-876C-56485DE04024.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018B-UL2018_MiniAODv2-v1/out_615.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260000/D492CBCC-062A-8F4F-9B53-DE9A64F257D7.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018B-UL2018_MiniAODv2-v1/out_510.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260000/7AF83D6A-1929-E346-AF53-38697CC1C364.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018B-UL2018_MiniAODv2-v1/out_154.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260000/DF8FC993-6FB3-FC4F-98A5-D373B2C509EF.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018B-UL2018_MiniAODv2-v1/out_513.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260000/DFF3DD18-8560-0A4E-B2B7-848E3738E216.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018B-UL2018_MiniAODv2-v1/out_512.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260001/DD851093-9C0A-C849-AB3C-873D5DB93DB3.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018B-UL2018_MiniAODv2-v1/out_948.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260001/5BBBECBD-FAAD-1A49-BDC0-5D53A31A281A.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018B-UL2018_MiniAODv2-v1/out_949.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260001/41AB3D41-BFA4-A443-97CC-5B05205EE297.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018B-UL2018_MiniAODv2-v1/out_946.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260001/0DBDD5A0-8C83-D54A-B990-F3355E3707C1.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018B-UL2018_MiniAODv2-v1/out_947.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260001/1D5B9CF5-24D5-7042-93C6-91BD4580A050.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018B-UL2018_MiniAODv2-v1/out_944.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260001/07EE32EF-A776-DE40-B5A0-4DBD73B9BFC4.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018B-UL2018_MiniAODv2-v1/out_945.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260001/FBD187D6-59E4-7A41-BEB1-4E3408284C21.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018B-UL2018_MiniAODv2-v1/out_942.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260001/369B1799-8E2A-4744-9A94-0E8577565250.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018B-UL2018_MiniAODv2-v1/out_943.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260001/F5CB44C6-99F1-AF48-9A4A-F81349C34041.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018B-UL2018_MiniAODv2-v1/out_940.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260001/9A0ECEB3-C4CC-A641-A5BE-D4BD8A0C28B0.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018B-UL2018_MiniAODv2-v1/out_941.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260002/30C5D4D5-801E-FF49-8987-B1FB52AA2508.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018B-UL2018_MiniAODv2-v1/out_1191.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260000/F11BA9C5-C492-774A-A8B3-C952325D57E3.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018B-UL2018_MiniAODv2-v1/out_345.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260000/2604885D-9B14-3644-8E7A-0DACC7E0D753.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018B-UL2018_MiniAODv2-v1/out_689.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260000/67A22344-4ADE-9544-BAFB-98773A7FCCB4.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018B-UL2018_MiniAODv2-v1/out_688.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260000/FB3A3607-E296-D448-9506-BB918195921E.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018B-UL2018_MiniAODv2-v1/out_157.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260002/671C6B58-63D6-4B42-AD0E-86C22EE43BF9.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018B-UL2018_MiniAODv2-v1/out_1190.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260000/99FE7354-5DD7-3340-8AE5-CDEEE92DF8B9.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018B-UL2018_MiniAODv2-v1/out_685.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260000/48B8643A-8573-F64D-BDFD-93D13B79030C.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018B-UL2018_MiniAODv2-v1/out_684.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260000/BDAB271C-7D56-C444-B8F7-AE75BD015399.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018B-UL2018_MiniAODv2-v1/out_687.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260000/7234567A-3E86-3B43-9625-CCA7FF9BD710.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018B-UL2018_MiniAODv2-v1/out_686.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260000/BE8BD2B4-8A54-AF45-9049-055EFC0DF814.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018B-UL2018_MiniAODv2-v1/out_681.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260000/BDFAED0C-9934-484C-A837-6E5559794466.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018B-UL2018_MiniAODv2-v1/out_680.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260000/97267249-FA8A-3C4B-B004-B17E0D1FEADE.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018B-UL2018_MiniAODv2-v1/out_683.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260000/2F3D74D6-5B50-394F-A2AA-5AAC50C789F4.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018B-UL2018_MiniAODv2-v1/out_682.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260001/EE21F3EB-3A4A-E542-BA45-5F5DA5979296.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018B-UL2018_MiniAODv2-v1/out_458.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260001/D0BAB317-57E6-C542-9396-360C14F8A5A9.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018B-UL2018_MiniAODv2-v1/out_459.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260001/5A0012BB-865C-6E4F-966F-275F4389BA69.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018B-UL2018_MiniAODv2-v1/out_621.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260000/5B8FBC75-BAA6-164E-9327-414E26D2B4C2.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018B-UL2018_MiniAODv2-v1/out_133.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260000/1E73B4FB-195A-934F-A733-50A0E06695EB.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018B-UL2018_MiniAODv2-v1/out_132.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260000/7FF482BE-0EC4-0546-A8FD-9D21297B8009.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018B-UL2018_MiniAODv2-v1/out_131.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260000/5B55267A-60A5-E64D-B54F-43ACC2E8C3AC.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018B-UL2018_MiniAODv2-v1/out_130.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260000/819D7907-C663-CD40-BEC5-22D16F74EEFC.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018B-UL2018_MiniAODv2-v1/out_137.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260000/86515125-1D33-7546-A727-7B083CC2A126.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018B-UL2018_MiniAODv2-v1/out_136.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260000/D94DA339-5E2D-AB47-90E3-EE22F43109CF.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018B-UL2018_MiniAODv2-v1/out_135.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260000/BD17DF7B-87D4-1E48-B120-35D952A6B670.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018B-UL2018_MiniAODv2-v1/out_134.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260000/D945B60A-BFA7-AB4C-8F9F-AC85FBB0103A.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018B-UL2018_MiniAODv2-v1/out_494.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260000/1DF0E922-F968-DB4F-971C-0C633357CCEC.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018B-UL2018_MiniAODv2-v1/out_495.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260000/BDBE0CA3-CFBC-2945-9C18-2DBEFFC15524.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018B-UL2018_MiniAODv2-v1/out_139.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260000/173932BF-BB45-8942-B24D-D6E265B6E238.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018B-UL2018_MiniAODv2-v1/out_138.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260000/FD4310B1-2326-4441-BD5B-316CE321F595.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018B-UL2018_MiniAODv2-v1/out_490.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260000/B049D99C-CDEE-AD42-80F1-D174FB4FACC0.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018B-UL2018_MiniAODv2-v1/out_491.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260000/9C755E79-479A-144D-8A29-A99D53082699.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018B-UL2018_MiniAODv2-v1/out_492.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260000/B4E6BA36-80F9-E143-81AA-196C06ADB22A.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018B-UL2018_MiniAODv2-v1/out_493.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260001/22731C3A-2448-B34A-8A17-B9DF14D2CF10.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018B-UL2018_MiniAODv2-v1/out_24.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260001/9C87C17C-44C2-4F45-A0FE-E19D5FBA6F53.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018B-UL2018_MiniAODv2-v1/out_25.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260001/21763786-2176-2348-89B2-B3547A324774.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018B-UL2018_MiniAODv2-v1/out_26.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260001/A230FF51-5135-934F-96E4-606E04320675.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018B-UL2018_MiniAODv2-v1/out_27.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260000/CAD108C7-C8C1-2E4A-8F5E-A862B2CCF071.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018B-UL2018_MiniAODv2-v1/out_20.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260000/F2ADF4D0-1447-BD44-B4CC-CA96619F691C.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018B-UL2018_MiniAODv2-v1/out_21.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260000/E2E8E945-D169-7442-BE89-A23BB85C2C5A.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018B-UL2018_MiniAODv2-v1/out_22.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260001/95B538C6-AD7F-1F4E-987C-2F0B64F9614D.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018B-UL2018_MiniAODv2-v1/out_23.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260000/A8A63F19-E3BB-3A4F-8F8D-A011DE257B80.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018B-UL2018_MiniAODv2-v1/out_1029.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260000/B20AAF8C-7EBA-3548-8508-3C6DCF40CF80.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018B-UL2018_MiniAODv2-v1/out_1220.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260001/03F34174-C57A-7640-9A4A-5BBBC71D754B.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018B-UL2018_MiniAODv2-v1/out_28.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260001/4F801FC7-3BF7-9F45-908C-30945E513089.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018B-UL2018_MiniAODv2-v1/out_29.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260000/A62DA97D-A84C-9145-B49A-E8B9C13FC2CD.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018B-UL2018_MiniAODv2-v1/out_820.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260000/E1813751-0E79-4348-9A01-CD7A636E6A16.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018B-UL2018_MiniAODv2-v1/out_407.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260000/A97FA1DE-31F0-914B-9816-B50BBA537B3A.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018B-UL2018_MiniAODv2-v1/out_406.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260000/2A1B64E1-210F-274C-8FE0-EB8A4D5E537A.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018B-UL2018_MiniAODv2-v1/out_405.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260001/9AF774B6-D0DA-414C-AD12-FAC76BAA4FCF.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018B-UL2018_MiniAODv2-v1/out_404.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260001/E2BA8139-8324-A640-879A-445E8016305F.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018B-UL2018_MiniAODv2-v1/out_403.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260001/9120F0C5-9CB5-E446-99E4-1B3B07E0EBA2.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018B-UL2018_MiniAODv2-v1/out_402.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260001/AE9501CA-F830-D049-B046-424D7DD187CB.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018B-UL2018_MiniAODv2-v1/out_401.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260001/E7AA34ED-699A-4146-9D97-0A30E7A581E3.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018B-UL2018_MiniAODv2-v1/out_400.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260001/E10B944A-F03C-9D42-8251-692A19664838.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018B-UL2018_MiniAODv2-v1/out_933.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260001/DE8E7E52-4482-4D41-A4CF-0364ED35FD0B.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018B-UL2018_MiniAODv2-v1/out_932.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260001/242627C9-412A-3146-84E3-5E0958070DF4.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018B-UL2018_MiniAODv2-v1/out_931.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260001/4893A881-26AE-104B-8D0A-E0A809CD3161.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018B-UL2018_MiniAODv2-v1/out_930.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260001/E19D16AA-E9B1-B644-9F99-FA13D7FEEADA.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018B-UL2018_MiniAODv2-v1/out_937.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260001/D6E228D9-8680-CF4C-926E-777BF6E2BBE5.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018B-UL2018_MiniAODv2-v1/out_452.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260000/5CC4E087-9754-CB41-9E18-64A37320F19F.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018B-UL2018_MiniAODv2-v1/out_409.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260000/63113B3C-5072-DF4C-9410-33882F53B1AD.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018B-UL2018_MiniAODv2-v1/out_408.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260001/A95A94DD-8707-CE44-BB15-6315C66F7174.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018B-UL2018_MiniAODv2-v1/out_453.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260001/960D9918-8590-5547-AE2B-BA573FB221EA.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018B-UL2018_MiniAODv2-v1/out_414.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260001/8D8B52D9-F487-3B41-B80B-F1154CF7CAF9.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018B-UL2018_MiniAODv2-v1/out_455.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260001/54448FA8-1E21-F246-93B3-0ABECE2A5047.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018B-UL2018_MiniAODv2-v1/out_456.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260002/202FB771-F5A8-A34D-9DC1-26B6AEC763BE.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018B-UL2018_MiniAODv2-v1/out_1118.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260001/AC6FEE68-8171-4E49-A100-18F1ED9EA09F.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018B-UL2018_MiniAODv2-v1/out_457.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260001/DE5A6691-90D3-8D41-A106-598568B2B596.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018B-UL2018_MiniAODv2-v1/out_183.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260000/055A9894-33E1-9840-8CB1-4E0325F6D6EC.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018B-UL2018_MiniAODv2-v1/out_849.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260000/68461CEF-53CF-A44E-9D90-B269BE7045DE.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018B-UL2018_MiniAODv2-v1/out_415.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260001/CA48358E-C635-B044-9201-2AEED56E5493.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018B-UL2018_MiniAODv2-v1/out_737.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260000/4C2E22F6-88AE-3447-AF6C-1D3A9B1E3723.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018B-UL2018_MiniAODv2-v1/out_379.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260000/497CE791-9F8A-BA4C-8F96-0AD5E1F89DAB.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018B-UL2018_MiniAODv2-v1/out_378.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260000/40EF37A7-5071-3546-B31F-BF7C9F430F52.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018B-UL2018_MiniAODv2-v1/out_228.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260000/F4D9E9BB-DFE5-0A43-8509-AE3F73A86F42.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018B-UL2018_MiniAODv2-v1/out_829.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260000/D265F580-53E8-674F-B075-80D6B094FE47.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018B-UL2018_MiniAODv2-v1/out_828.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260001/362A6B20-7C66-9349-B8E8-13E9C8E8ADEC.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018B-UL2018_MiniAODv2-v1/out_44.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260000/1A692C81-4AE3-634D-B520-8562CACD6A14.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018B-UL2018_MiniAODv2-v1/out_371.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260000/B402F7A4-EDAA-1F4A-A8A4-3CFFC261F767.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018B-UL2018_MiniAODv2-v1/out_370.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260000/FADAE81F-6742-6641-9E0A-FF95ED6F8316.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018B-UL2018_MiniAODv2-v1/out_373.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260000/4A9A8377-18C0-4E4D-AACF-6B2E037F72E1.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018B-UL2018_MiniAODv2-v1/out_372.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260000/B59AB654-E621-5E42-9D0B-D52798C05D1D.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018B-UL2018_MiniAODv2-v1/out_375.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260000/0214E98E-3E67-C04A-ADCA-96CCBD9654FB.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018B-UL2018_MiniAODv2-v1/out_374.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260000/6ED9B964-3EF2-E044-BEEA-EC4871C84900.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018B-UL2018_MiniAODv2-v1/out_377.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260000/646EC525-697C-2C45-9ED4-185872852A1C.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018B-UL2018_MiniAODv2-v1/out_376.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260000/0DEF7BF8-57A6-D343-96D9-22C0B7EA7D04.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018B-UL2018_MiniAODv2-v1/out_1177.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260000/B9568F16-AC01-D342-85BD-8F0F76285FCE.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018B-UL2018_MiniAODv2-v1/out_708.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260000/3F50DE30-2493-5249-808D-96A95B55A14D.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018B-UL2018_MiniAODv2-v1/out_709.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260002/C599FA12-817D-7D41-86EE-645FFE09B7DC.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018B-UL2018_MiniAODv2-v1/out_1176.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260000/E5BFD482-0CBC-A043-8081-FA224CC9B91B.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018B-UL2018_MiniAODv2-v1/out_704.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260000/6A8DFD01-DF1A-9B47-8BE2-CEE0D253A5C4.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018B-UL2018_MiniAODv2-v1/out_705.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260000/BA8E4B66-2367-5946-A1CE-4AAC39DACFCB.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018B-UL2018_MiniAODv2-v1/out_706.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260000/8096F60C-983A-CD46-B289-9C923693825B.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018B-UL2018_MiniAODv2-v1/out_707.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260000/91FB2A62-0646-FC4E-BEF3-34DE1E24FAB1.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018B-UL2018_MiniAODv2-v1/out_700.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260002/80E94DCF-1AD0-4542-B466-D75545B0EF18.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018B-UL2018_MiniAODv2-v1/out_1175.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260000/E085C2F9-DF71-044D-ADAD-82CFB3013C7B.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018B-UL2018_MiniAODv2-v1/out_702.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260000/5CBBDCCA-2333-7F44-AD7F-65DE586D7872.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018B-UL2018_MiniAODv2-v1/out_703.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260001/B0297D84-9341-5444-8BB8-70CE72C709D5.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018B-UL2018_MiniAODv2-v1/out_393.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260001/79E030A9-5510-8047-A72F-9EECBA2F78DC.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018B-UL2018_MiniAODv2-v1/out_392.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260000/F4DF6193-2F5C-3D41-8C50-24B1299DED1D.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018B-UL2018_MiniAODv2-v1/out_88.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260001/83BEE168-E231-7B40-86C4-385208A3F9A6.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018B-UL2018_MiniAODv2-v1/out_619.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260001/DB07EB3D-EEA1-7A4A-9663-B8501CEFB69E.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018B-UL2018_MiniAODv2-v1/out_397.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260001/3EA12EBC-EA16-D944-BC54-CF051757E2DA.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018B-UL2018_MiniAODv2-v1/out_396.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260001/03F53B03-DB1F-E744-A559-A0A37E9D9377.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018B-UL2018_MiniAODv2-v1/out_395.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260001/7729E072-8CE9-4144-9554-8304ED5AE27F.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018B-UL2018_MiniAODv2-v1/out_394.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260000/EA85DD1A-1F80-BD46-8ECA-6B71CDAA0082.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018B-UL2018_MiniAODv2-v1/out_82.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260000/78310E81-881F-FA40-B0DB-6FE59142C075.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018B-UL2018_MiniAODv2-v1/out_83.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260000/E939F198-2496-D94F-8A88-BA5E698F998E.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018B-UL2018_MiniAODv2-v1/out_80.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260000/906C7615-77DA-BA4F-BEAF-4667AC754767.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018B-UL2018_MiniAODv2-v1/out_81.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260000/A1B6BD9C-7DFC-374E-B8CD-D583289CE9E5.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018B-UL2018_MiniAODv2-v1/out_86.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260000/B897E92A-9ADB-C04F-8D00-C72BE2AA014D.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018B-UL2018_MiniAODv2-v1/out_87.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260000/59A46A44-D4D2-894C-AAFE-CF865A1209EA.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018B-UL2018_MiniAODv2-v1/out_84.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260000/B00209AA-CE0A-A04B-B962-95A4A0ABFCC5.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018B-UL2018_MiniAODv2-v1/out_85.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260001/782D8CB6-87F4-0940-8A9E-74FA251E43B6.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018B-UL2018_MiniAODv2-v1/out_797.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260001/ED0BBE8B-BA34-5A45-9A31-FE632B9C4E4C.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018B-UL2018_MiniAODv2-v1/out_796.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260001/57F95325-2A46-2148-8E51-E3EB55A69C3C.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018B-UL2018_MiniAODv2-v1/out_795.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260001/AC539F12-A9CF-6446-989D-170980F2236C.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018B-UL2018_MiniAODv2-v1/out_794.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260001/C39095C4-2670-3A46-8D3D-E65A3DDF200D.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018B-UL2018_MiniAODv2-v1/out_793.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260001/92D940EF-9B3D-CB42-ABFA-1307D34A6BEC.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018B-UL2018_MiniAODv2-v1/out_792.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260001/52B9C730-5A39-114D-B2FB-A00A0DF73513.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018B-UL2018_MiniAODv2-v1/out_791.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260001/6AC496FD-DD5E-4149-86CC-4FBF5DB13AFE.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018B-UL2018_MiniAODv2-v1/out_790.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260002/4339A5BA-5709-2D41-8720-74389F9F0186.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018B-UL2018_MiniAODv2-v1/out_1170.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260000/1807016D-5C02-F045-B8CD-2939BA1CA9EA.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018B-UL2018_MiniAODv2-v1/out_749.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260001/2F3299AE-7A91-054A-999E-F3DD06382141.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018B-UL2018_MiniAODv2-v1/out_799.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260001/D5BE14B9-946D-9C46-A241-3EDDDA54D56C.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018B-UL2018_MiniAODv2-v1/out_798.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260001/763DB538-EAF9-9345-AE5D-885E4999B007.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018B-UL2018_MiniAODv2-v1/out_7.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260001/FB28EF3E-34DC-0C4D-A69A-159CDCC66701.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018B-UL2018_MiniAODv2-v1/out_170.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260000/E69F9E30-9FFA-B142-81C0-033ACC14F18A.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018B-UL2018_MiniAODv2-v1/out_586.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260000/96D4E201-81BD-2B45-A5F6-BE3B781C53F2.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018B-UL2018_MiniAODv2-v1/out_587.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260002/D5F53F02-8F7A-AD4D-8A0A-7143E0D151D8.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018B-UL2018_MiniAODv2-v1/out_1139.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260000/98BDFBEE-00E1-0C43-AA64-2120F63C5F4B.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018B-UL2018_MiniAODv2-v1/out_585.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260000/27A226FD-3549-334D-B81E-A1FE0E86AE25.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018B-UL2018_MiniAODv2-v1/out_582.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260000/8634542F-03F8-2F4B-BA85-A7C50EF999AA.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018B-UL2018_MiniAODv2-v1/out_583.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260000/0C012F55-2E02-F847-ACD0-87145EBECADE.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018B-UL2018_MiniAODv2-v1/out_580.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260000/DCE3F740-FAF1-DB48-B0E6-6E7EA1D05643.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018B-UL2018_MiniAODv2-v1/out_581.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260002/21957F9C-3C79-4D40-86FF-61D60425E690.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018B-UL2018_MiniAODv2-v1/out_1133.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260002/B30F5E74-D93C-464F-901A-A8F26FD37ED8.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018B-UL2018_MiniAODv2-v1/out_1132.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260002/BF221A84-2FAF-384A-9555-55C250F7F551.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018B-UL2018_MiniAODv2-v1/out_1131.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260002/D5492A04-3CB3-444B-8464-E57CB5B435EC.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018B-UL2018_MiniAODv2-v1/out_1130.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260002/D4E93744-FF90-CC41-9730-0D09031F7843.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018B-UL2018_MiniAODv2-v1/out_1137.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260002/F77152B6-626A-B74E-994C-853E07A3AF7C.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018B-UL2018_MiniAODv2-v1/out_1136.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260002/A068A306-78B9-0C45-9589-52BE6DE2AA49.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018B-UL2018_MiniAODv2-v1/out_1135.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260000/472C9709-E4A7-9148-81A4-F0E3228FB60D.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018B-UL2018_MiniAODv2-v1/out_589.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260001/9024003F-A4BC-C541-9352-0BF63F19D222.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018B-UL2018_MiniAODv2-v1/out_245.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260001/61A3B0F0-6997-A642-AE08-E4DBE6706126.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018B-UL2018_MiniAODv2-v1/out_244.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260001/C96C7626-C827-D843-A091-AC984E015274.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018B-UL2018_MiniAODv2-v1/out_247.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260001/5BB0E274-74D9-9C41-9BA2-5269F2E056BF.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018B-UL2018_MiniAODv2-v1/out_246.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260001/2FAEC95F-F893-2D4A-817E-500A03C233BB.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018B-UL2018_MiniAODv2-v1/out_241.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260000/712B5BE5-42B4-1243-8600-F87F4CDFCECA.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018B-UL2018_MiniAODv2-v1/out_614.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260001/BF9024A1-7E82-EF46-9CEC-32D73462FF8E.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018B-UL2018_MiniAODv2-v1/out_243.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260001/96D8D337-6217-B84E-A662-3D51EAEFBF32.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018B-UL2018_MiniAODv2-v1/out_242.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260002/4743F41C-92BA-1C41-A28F-73BDC4C2A49D.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018B-UL2018_MiniAODv2-v1/out_1178.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260001/BB5B415F-2965-0448-99CA-88D48E1258D6.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018B-UL2018_MiniAODv2-v1/out_249.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260001/BEB10BA9-FBEC-A84B-99CF-32E28792DD3B.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018B-UL2018_MiniAODv2-v1/out_248.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260001/C3B56093-177D-6A43-BE1D-C6052F92B47F.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018B-UL2018_MiniAODv2-v1/out_972.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260000/1240568D-33F7-884B-B3D2-6BAE1534A2A1.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018B-UL2018_MiniAODv2-v1/out_418.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260000/5CCFDE37-EE66-F241-BD58-037D8E277593.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018B-UL2018_MiniAODv2-v1/out_224.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260000/18B0F3C0-CD25-0B43-A733-94B0D623710A.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018B-UL2018_MiniAODv2-v1/out_639.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260000/36A582AF-11F1-4C4D-8C5B-8B9EAFCE82BC.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018B-UL2018_MiniAODv2-v1/out_419.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260000/93A88904-5B77-054B-B991-765FC75C1D8B.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018B-UL2018_MiniAODv2-v1/out_519.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260000/1F8E7AD4-3C3D-5146-9E41-324C5C943BC1.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018B-UL2018_MiniAODv2-v1/out_518.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260002/EC4EC0B1-9616-1047-9941-D09183958B32.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018B-UL2018_MiniAODv2-v1/out_1009.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260000/DFF73C66-8B28-F645-8D79-09D253053B27.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018B-UL2018_MiniAODv2-v1/out_1008.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260000/3ADB7F76-4BA9-1A4B-8F27-3287C2428129.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018B-UL2018_MiniAODv2-v1/out_1007.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260002/25CB57CB-FB7E-7047-B676-95FD59344E5B.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018B-UL2018_MiniAODv2-v1/out_1006.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260002/B822258F-9847-C346-B44D-70C1544D38FD.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018B-UL2018_MiniAODv2-v1/out_1005.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260000/8A1A4F09-0F6D-A647-93E2-9583A7604004.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018B-UL2018_MiniAODv2-v1/out_1004.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260000/D0B566BF-5EE4-C147-B68A-520068BDAA0B.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018B-UL2018_MiniAODv2-v1/out_515.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260002/F3A35932-D9E5-3B40-9BDD-0E2B7DED3CF6.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018B-UL2018_MiniAODv2-v1/out_1002.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260000/414D719F-A8E3-5C43-B141-AAA6816CBEA0.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018B-UL2018_MiniAODv2-v1/out_517.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260002/E4FA27FC-9C25-7047-B34F-EC5AA4DD1AC6.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018B-UL2018_MiniAODv2-v1/out_1000.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260001/B5D84DAB-DBBE-664B-83E1-309B06872F66.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018B-UL2018_MiniAODv2-v1/out_623.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260001/9982DE71-4DBB-304C-8AF2-BB7EE52BA720.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018B-UL2018_MiniAODv2-v1/out_622.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260000/0A9E7AD1-9943-F049-9E77-4659E11836AF.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018B-UL2018_MiniAODv2-v1/out_1225.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260000/72FB1DD2-F87C-AD49-8B6A-8DF69973A05C.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018B-UL2018_MiniAODv2-v1/out_514.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260001/8569BBB6-1E80-5143-95AB-A70636237F90.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018B-UL2018_MiniAODv2-v1/out_627.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260001/7C6D8A8D-05B0-B64A-911B-BCB180E46D76.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018B-UL2018_MiniAODv2-v1/out_626.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260001/9CEB3448-6AEA-7F4E-A38C-4EF2B4F80659.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018B-UL2018_MiniAODv2-v1/out_625.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260001/9CA12ED8-AA4E-B54B-BF8F-8C10A2B7263B.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018B-UL2018_MiniAODv2-v1/out_624.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260000/CF66D82C-B921-4A42-BBC1-BD8FCD003FFC.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018B-UL2018_MiniAODv2-v1/out_450.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260001/05D02B3F-5DE2-074A-85DA-737752A41670.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018B-UL2018_MiniAODv2-v1/out_451.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260001/E34C5C53-397B-EE48-AB9A-1A19CE5F4B09.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018B-UL2018_MiniAODv2-v1/out_629.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260001/4FD3EFC1-63D3-B444-9234-18C0CDCCA1E7.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018B-UL2018_MiniAODv2-v1/out_628.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260001/7FE273CB-8A00-8547-8676-79E985436EDF.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018B-UL2018_MiniAODv2-v1/out_454.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260002/879A0FB0-A95D-9941-81E8-A68DC70D0D45.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018B-UL2018_MiniAODv2-v1/out_1085.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260002/4B1209EF-A0DC-0E4D-959B-9DB5AD94913E.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018B-UL2018_MiniAODv2-v1/out_1229.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260002/6E948BB6-F655-A941-A0FF-9BEEA8599229.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018B-UL2018_MiniAODv2-v1/out_1228.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260002/D1BE9439-8DE2-4549-9EC2-381A864EB84E.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018B-UL2018_MiniAODv2-v1/out_1084.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260000/5BF39409-0F23-1643-AEBB-D3B51E57AD38.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018B-UL2018_MiniAODv2-v1/out_179.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260000/63AFBC43-7184-694A-BCA7-C64AC5CADC45.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018B-UL2018_MiniAODv2-v1/out_178.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260000/46BD90FD-57EA-C94B-B457-CC1B2B645DF0.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018B-UL2018_MiniAODv2-v1/out_177.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260000/9024CE0B-E133-1E43-B50A-DD33B3AAA444.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018B-UL2018_MiniAODv2-v1/out_199.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260000/76686490-1C9C-334B-98E0-12DC462FB431.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018B-UL2018_MiniAODv2-v1/out_175.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260000/B68672C8-E6B7-B94F-87B1-3F4D16E8E27F.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018B-UL2018_MiniAODv2-v1/out_174.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260001/AE2554F0-9613-BB41-85AC-A8FB6E38E4B2.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018B-UL2018_MiniAODv2-v1/out_173.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260001/A29E0159-3785-794C-BD1B-34BC52DBCCD2.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018B-UL2018_MiniAODv2-v1/out_172.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260001/7B936AD6-D5A6-DF4A-B08C-41D9983B5B33.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018B-UL2018_MiniAODv2-v1/out_171.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260000/61BDE386-22DA-8E41-811A-8C33706B3A80.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018B-UL2018_MiniAODv2-v1/out_594.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260001/94427552-10E1-0E4A-AC2E-4F7B13205AC5.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018B-UL2018_MiniAODv2-v1/out_977.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260001/69B0D11C-160A-014B-B3BF-06A73D37F3BD.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018B-UL2018_MiniAODv2-v1/out_182.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260001/EAE1F753-B630-6941-A1AB-EEF9231C596F.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018B-UL2018_MiniAODv2-v1/out_975.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260001/342242A1-4B1F-8548-895C-883CBBBED62D.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018B-UL2018_MiniAODv2-v1/out_974.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260001/65AE17DA-88D1-6E4E-919C-DA07753006A8.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018B-UL2018_MiniAODv2-v1/out_973.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260002/3F425A9C-07D5-0346-8595-86527E3A66FA.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018B-UL2018_MiniAODv2-v1/out_1081.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260000/E74C8BAC-2C71-564A-BD99-C954DED80048.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018B-UL2018_MiniAODv2-v1/out_971.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260000/03DA1C3D-39C0-AE49-A21C-D1A4A16AE3E4.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018B-UL2018_MiniAODv2-v1/out_970.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260002/B22D9E76-55A0-BD45-B0B2-0A8521586838.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018B-UL2018_MiniAODv2-v1/out_1080.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260000/D16D1350-C1EE-7F44-8781-B383875F5DFC.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018B-UL2018_MiniAODv2-v1/out_180.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260001/C8F35381-CD7D-A148-8BFC-83483C7C417D.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018B-UL2018_MiniAODv2-v1/out_979.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260001/FC1C69AF-EA04-3540-8CE5-4DB9BF7B1FD0.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018B-UL2018_MiniAODv2-v1/out_978.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260000/7EB71D93-C9AF-BC49-86CB-DE7FE4F60E44.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018B-UL2018_MiniAODv2-v1/out_656.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260002/0762A984-6578-F04C-AC0D-9AF64E49739E.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018B-UL2018_MiniAODv2-v1/out_1120.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260000/D81FBEEF-7D33-0D47-ABF1-21EB8CA2C34B.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018B-UL2018_MiniAODv2-v1/out_654.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260000/1D8586A5-C183-C84E-B538-178BFBFC1551.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018B-UL2018_MiniAODv2-v1/out_181.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260001/1455E21C-1B19-D54F-8CD8-96F4B3B72FF1.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018B-UL2018_MiniAODv2-v1/out_186.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260001/AD8624FE-3F24-7D46-9903-65DC3B00E151.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018B-UL2018_MiniAODv2-v1/out_187.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260001/24AB71CE-1A14-A64A-8BBE-BF3F7C4DF796.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018B-UL2018_MiniAODv2-v1/out_184.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260001/5C11B505-A5B4-FC4E-BC71-3F654EDE47A3.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018B-UL2018_MiniAODv2-v1/out_185.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260000/ED7DB24A-1948-A24A-87B5-8D643747C446.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018B-UL2018_MiniAODv2-v1/out_652.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260001/821E413E-2728-BD4C-9267-FF7838BED271.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018B-UL2018_MiniAODv2-v1/out_188.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260001/18B00059-CDE3-2547-B244-9E5AE16780F9.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018B-UL2018_MiniAODv2-v1/out_189.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260002/9930C3C4-50B7-3B44-952D-9138DA9DC3E6.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018B-UL2018_MiniAODv2-v1/out_1122.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260000/022F328B-9526-844E-A003-AFAE3E0A5EC5.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018B-UL2018_MiniAODv2-v1/out_658.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260000/25DEE84D-6703-2545-B9EB-718860022738.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018B-UL2018_MiniAODv2-v1/out_516.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260002/C1B8B860-8E25-994C-B0C0-12D86ACD260B.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018B-UL2018_MiniAODv2-v1/out_1123.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260000/E3021C4B-C937-BC4A-BD9D-10FDD5274918.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018B-UL2018_MiniAODv2-v1/out_650.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260002/8F7F25CE-4885-8042-9C9B-08427FBE96EB.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018B-UL2018_MiniAODv2-v1/out_1124.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260000/BA75ABAC-0ECA-8646-A86C-8FA2ACB8AFE6.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018B-UL2018_MiniAODv2-v1/out_651.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260002/0627880F-9333-7840-99D4-C27DF5D81C27.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018B-UL2018_MiniAODv2-v1/out_1125.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260000/E17C24B7-AFAA-E54F-BF62-A41DEF457A3E.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018B-UL2018_MiniAODv2-v1/out_1089.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260000/C2C8E346-7ADC-AD42-9EA1-019E102CD69E.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018B-UL2018_MiniAODv2-v1/out_1088.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260002/4E31D0CB-EE16-8841-A1B5-FF948558DD6B.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018B-UL2018_MiniAODv2-v1/out_1227.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260001/D301887C-8FEA-9642-9AF1-BD1458C98B2E.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018B-UL2018_MiniAODv2-v1/out_11.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260001/AF028E56-538F-7447-AA9A-96426CC15F9E.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018B-UL2018_MiniAODv2-v1/out_10.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260001/5F982390-0C96-3B46-88A0-9F5857C092AC.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018B-UL2018_MiniAODv2-v1/out_13.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260001/F5DC723C-B066-1942-A22B-531BE4D92896.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018B-UL2018_MiniAODv2-v1/out_12.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260002/C7FA76CA-85C7-9448-988E-0B59E7BDDC1E.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018B-UL2018_MiniAODv2-v1/out_15.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260001/8BF53C31-01D6-E34B-9F2B-FD0A9DE35F08.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018B-UL2018_MiniAODv2-v1/out_14.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260002/C2E103E0-06AF-8D4D-B321-AEDACB213A84.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018B-UL2018_MiniAODv2-v1/out_17.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260002/C929B577-6AEA-B94F-ACCB-72636FFC0EA2.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018B-UL2018_MiniAODv2-v1/out_16.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260000/8EF6AE95-B91D-8F45-B419-9B990954EF2E.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018B-UL2018_MiniAODv2-v1/out_19.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260002/A1449267-1B5F-EA4C-8FD1-6F056AA9F9A6.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018B-UL2018_MiniAODv2-v1/out_18.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260001/74AC7063-8D05-2947-B008-D87ED85DA9FA.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018B-UL2018_MiniAODv2-v1/out_863.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260001/7355E9C3-686B-0543-8E8C-D0E85CE75FCB.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018B-UL2018_MiniAODv2-v1/out_862.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260001/7F9DAFC8-3B7A-4546-99A8-B74AAFBBD70B.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018B-UL2018_MiniAODv2-v1/out_865.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260001/D6ACC508-2948-134A-9A29-AFA862765EEB.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018B-UL2018_MiniAODv2-v1/out_864.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260001/384E21C0-1545-0846-B45A-411BE16CE8D9.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018B-UL2018_MiniAODv2-v1/out_867.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260001/11EA7045-A98E-AC4C-9972-BE50C26CD044.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018B-UL2018_MiniAODv2-v1/out_866.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260000/B4ADB994-A5E3-C043-8551-88381B7DA351.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018B-UL2018_MiniAODv2-v1/out_354.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260000/6D629995-A87F-3F49-BD81-701E756B75DA.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018B-UL2018_MiniAODv2-v1/out_1210.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260000/3C00A17E-9DF4-874D-82C2-576FEEE1D78F.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018B-UL2018_MiniAODv2-v1/out_659.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260000/CFC37539-0B49-A54E-88D9-4FC051E49FA9.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018B-UL2018_MiniAODv2-v1/out_1226.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260001/2AC0CC61-3EB8-E748-A036-D3AF24AC59C1.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018B-UL2018_MiniAODv2-v1/out_883.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260001/F8EDEA83-27E8-5F40-82B0-27F0420425E1.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018B-UL2018_MiniAODv2-v1/out_882.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260001/EFE4FAD5-002B-B141-A9B0-5965D31CEF9C.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018B-UL2018_MiniAODv2-v1/out_881.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260001/77C231DB-6512-2541-AA9D-2F3E089472B0.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018B-UL2018_MiniAODv2-v1/out_880.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260001/F5D89E8B-7BF6-6144-ABD8-7747BBC78679.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018B-UL2018_MiniAODv2-v1/out_887.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260001/071FB6CC-1275-204C-B656-F49587C052B6.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018B-UL2018_MiniAODv2-v1/out_886.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260001/8BE5436E-A205-1F41-9F52-333EEE7D5F54.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018B-UL2018_MiniAODv2-v1/out_885.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260001/7FC713D7-2C05-4846-9AD1-1DD3318967F4.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018B-UL2018_MiniAODv2-v1/out_884.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260001/4870F03C-1700-3E49-BA63-6BD321D23B90.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018B-UL2018_MiniAODv2-v1/out_889.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260001/5470B072-FB7A-C94F-8AA7-32B9FC782D06.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018B-UL2018_MiniAODv2-v1/out_888.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260001/3C38751C-4C13-964D-AB98-B3D4602C4CAD.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018B-UL2018_MiniAODv2-v1/out_116.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260001/2A67340B-AD87-F548-B8AE-9C3D98D78EBD.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018B-UL2018_MiniAODv2-v1/out_45.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260000/B372F092-7A0B-3E4E-BC73-1EAF63E64FD8.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018B-UL2018_MiniAODv2-v1/out_657.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260000/32FE4876-8DB7-0D40-8212-755D3B101A94.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018B-UL2018_MiniAODv2-v1/out_356.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260002/6BACAE7A-0DE3-3942-8638-669B67E71661.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018B-UL2018_MiniAODv2-v1/out_1083.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260000/357F60B6-7F2A-2C40-8803-88CF88B73EB3.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018B-UL2018_MiniAODv2-v1/out_355.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260001/456EC380-1423-7746-BDF7-00EB4701D3EF.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018B-UL2018_MiniAODv2-v1/out_322.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260001/D5916545-40AF-B14C-8EFD-516509C52353.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018B-UL2018_MiniAODv2-v1/out_323.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260001/D1EA531F-870F-B344-A910-2921BEC1012D.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018B-UL2018_MiniAODv2-v1/out_320.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260001/9707C790-B2B9-7B4C-9638-91A6F775FAB5.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018B-UL2018_MiniAODv2-v1/out_321.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260001/923C053F-F957-4343-9071-535F6F4453D7.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018B-UL2018_MiniAODv2-v1/out_326.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260001/B6AB9D3A-DA10-D542-B6FA-CA2551DE1E9E.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018B-UL2018_MiniAODv2-v1/out_327.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260001/F68E3CD1-CBA8-B542-9C05-0A6D6CEB91DD.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018B-UL2018_MiniAODv2-v1/out_324.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260001/CEC95F59-9F10-3548-917F-10C0CC2503B5.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018B-UL2018_MiniAODv2-v1/out_325.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260002/E94F2E5C-8BD9-5148-BF6F-B4DEFFF10198.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018B-UL2018_MiniAODv2-v1/out_1224.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260001/279E8E57-5B52-8B4B-A563-628529431BA1.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018B-UL2018_MiniAODv2-v1/out_328.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260001/D49A84E4-FBDA-BF41-92D9-145F15D6142A.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018B-UL2018_MiniAODv2-v1/out_329.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260000/F4A2607B-9B91-514E-9711-3BE7467E4701.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018B-UL2018_MiniAODv2-v1/out_562.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260000/AC17295E-A3C8-E64A-8800-07E8B752B0B7.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018B-UL2018_MiniAODv2-v1/out_201.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260000/28FAB1C9-2705-7B4F-AC84-EE7B8C304D7C.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018B-UL2018_MiniAODv2-v1/out_200.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260000/DDE146CF-9420-5546-92B4-E3226CC9DCC7.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018B-UL2018_MiniAODv2-v1/out_203.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260000/EA953E25-64ED-574A-AD3E-C7ADA7A17EF6.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018B-UL2018_MiniAODv2-v1/out_202.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260000/E1ED76CF-1C39-324C-93D9-3020E567FB2F.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018B-UL2018_MiniAODv2-v1/out_205.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260000/6422AB25-D19E-BD4D-87A6-39FB80B1FBC2.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018B-UL2018_MiniAODv2-v1/out_204.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260000/215A3094-92AD-244B-8C00-C83B2317A146.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018B-UL2018_MiniAODv2-v1/out_207.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260000/2FA523BC-B07F-A844-8B8B-0E32991FA929.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018B-UL2018_MiniAODv2-v1/out_206.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260000/4F16FCE6-167A-4C4B-BC02-218EB787A4AB.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018B-UL2018_MiniAODv2-v1/out_209.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260000/C685CDB9-B7BB-DB48-B4C6-7C81A698AE73.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018B-UL2018_MiniAODv2-v1/out_208.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260000/343F945C-1278-3D4A-9626-E183499631C4.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018B-UL2018_MiniAODv2-v1/out_779.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260000/C908AE07-31F4-8E4F-94E9-30C5A45E433E.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018B-UL2018_MiniAODv2-v1/out_778.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260000/52CD55E6-BBD9-6B4F-8E3B-BE4ADEE33137.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018B-UL2018_MiniAODv2-v1/out_679.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260000/786AB91A-DB8D-4044-B4DC-9C54D9584267.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018B-UL2018_MiniAODv2-v1/out_77.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260000/B1139EFD-7E12-E14F-B39B-54C2221DB1D7.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018B-UL2018_MiniAODv2-v1/out_76.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260000/9AC7EBF9-6C53-9646-BB52-F6849BC49326.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018B-UL2018_MiniAODv2-v1/out_75.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260000/191BACB4-E862-1248-BBAA-0653B88F2838.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018B-UL2018_MiniAODv2-v1/out_74.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260000/66448078-6D4B-4F43-AE18-D0C8351B998B.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018B-UL2018_MiniAODv2-v1/out_73.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260000/E4E1996F-7715-2743-AE19-B0A3353A0E61.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018B-UL2018_MiniAODv2-v1/out_72.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260000/570CFFDC-A21A-1B4B-92D0-3A752D54C119.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018B-UL2018_MiniAODv2-v1/out_71.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260000/28409424-5A7B-7449-9CD9-6E1BCFF65B35.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018B-UL2018_MiniAODv2-v1/out_70.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260000/73F79514-CA8E-2543-8997-CDEACF85584A.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018B-UL2018_MiniAODv2-v1/out_655.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260000/B892C155-D356-8E4C-8B37-53C998D5CAB7.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018B-UL2018_MiniAODv2-v1/out_631.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260000/47C967CF-4C7C-9743-B197-1900293671D4.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018B-UL2018_MiniAODv2-v1/out_567.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260000/830A9C8C-8AC5-7B48-BCB8-E457726EBB16.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018B-UL2018_MiniAODv2-v1/out_79.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260000/8935DAC3-78A8-9A44-AE41-7D54E5F864B3.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018B-UL2018_MiniAODv2-v1/out_78.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260002/8B7533B1-10F9-F446-9A01-3227F51060C5.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018B-UL2018_MiniAODv2-v1/out_1043.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260002/72D3467B-498D-0B46-9523-E890E77B5C17.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018B-UL2018_MiniAODv2-v1/out_1042.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260002/61186132-6B31-FC4C-ABBA-B44469790DF2.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018B-UL2018_MiniAODv2-v1/out_1041.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260002/744485AC-FF5D-8740-BE9B-E964729088AD.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018B-UL2018_MiniAODv2-v1/out_1040.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260002/90ECB0B4-D368-734D-A410-8526BD2557EA.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018B-UL2018_MiniAODv2-v1/out_1047.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260002/FB28EE21-CD40-3246-A65E-E08D1808DA08.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018B-UL2018_MiniAODv2-v1/out_1046.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260000/81DE1CE8-8583-D446-8230-7A9D9FC63D1A.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018B-UL2018_MiniAODv2-v1/out_1045.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260002/1E748ADD-A022-9C4B-B569-3CBB28FF9AFB.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018B-UL2018_MiniAODv2-v1/out_1044.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260002/E0DC8710-66CA-7147-A118-D68C56C84D15.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018B-UL2018_MiniAODv2-v1/out_1049.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260002/64E17D47-F610-224D-8958-AE36F53D8FCC.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018B-UL2018_MiniAODv2-v1/out_1048.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260002/A215B0E5-64F7-CF46-8574-DE93A6377696.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018B-UL2018_MiniAODv2-v1/out_1222.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260000/9656625B-92FF-3B49-B306-568C2405357E.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018B-UL2018_MiniAODv2-v1/out_359.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260000/37744A4D-B62F-8B4D-AFB2-94A6C964D913.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018B-UL2018_MiniAODv2-v1/out_358.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260001/DB286F16-99AB-2E41-8E07-A0420E78882B.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018B-UL2018_MiniAODv2-v1/out_868.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260002/0D7CA607-D7EA-454B-B98E-81208DFEB703.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018B-UL2018_MiniAODv2-v1/out_1142.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260000/C1EADA4D-FB8D-3D4A-8D25-5EDB2C1B69CD.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018B-UL2018_MiniAODv2-v1/out_1143.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260002/F433000F-B6F6-0145-98A7-38E50B3DAAC5.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018B-UL2018_MiniAODv2-v1/out_1140.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260000/C19249E2-B238-054D-A8A5-0084B3573831.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018B-UL2018_MiniAODv2-v1/out_1141.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260000/3AF3D826-C17C-EF4C-B881-E7C4AD99E389.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018B-UL2018_MiniAODv2-v1/out_1146.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260000/4980F709-5014-BD4F-837E-F33B8098CA9B.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018B-UL2018_MiniAODv2-v1/out_1147.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260002/17729583-D809-8B47-95A1-DE16E8FA9BFA.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018B-UL2018_MiniAODv2-v1/out_1144.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260002/816969E0-C3D0-2B42-9CB4-FEB4D3299E47.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018B-UL2018_MiniAODv2-v1/out_1145.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260001/0556A82A-8B2B-CB49-8831-1CB06BE9DB91.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018B-UL2018_MiniAODv2-v1/out_667.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260000/DDFB677F-5C5A-EE4A-BC81-09F9F3EBA674.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018B-UL2018_MiniAODv2-v1/out_666.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260002/150C2386-9092-D448-BC31-D684E13268C5.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018B-UL2018_MiniAODv2-v1/out_1148.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260002/07FF2C59-4672-B044-B7E0-DF98545691D0.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018B-UL2018_MiniAODv2-v1/out_1149.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260000/228B6B04-7C26-D74A-994D-C3607320066A.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018B-UL2018_MiniAODv2-v1/out_663.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260000/1A45F06A-EE1B-E145-81A9-0E8DF9B772F8.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018B-UL2018_MiniAODv2-v1/out_662.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260000/89F57033-1E47-204D-83B2-D0D537713618.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018B-UL2018_MiniAODv2-v1/out_661.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260000/224A86EE-6B8F-5045-B620-2E281A891069.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018B-UL2018_MiniAODv2-v1/out_660.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260000/2F78C42B-7D2F-B649-9E3F-66B45D76D134.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018B-UL2018_MiniAODv2-v1/out_1221.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260000/A4AC01B9-71BA-C44C-ACB6-D19A23C4892A.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018B-UL2018_MiniAODv2-v1/out_215.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260000/AE1ACC74-43F9-024D-BB54-044A69C11806.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018B-UL2018_MiniAODv2-v1/out_653.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260000/6DC6FE5D-E7A7-5C42-B632-06A78C66E9E9.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018B-UL2018_MiniAODv2-v1/out_1037.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260000/A5B53D6B-1F60-E747-9D82-405CAD6D906A.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018B-UL2018_MiniAODv2-v1/out_692.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260000/69D96C3C-CFC3-D941-AFA3-FF6E8DD9C43E.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018B-UL2018_MiniAODv2-v1/out_693.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260000/69513B01-BEE3-A247-8C63-1D4959D09766.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018B-UL2018_MiniAODv2-v1/out_690.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260000/DB699901-2BF0-664D-B503-819C3664106B.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018B-UL2018_MiniAODv2-v1/out_691.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260000/0CD2330B-0515-B243-992F-DEC5728F1651.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018B-UL2018_MiniAODv2-v1/out_696.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260000/69B3A681-F337-6E42-AC3B-92DB26BABE88.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018B-UL2018_MiniAODv2-v1/out_697.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260000/5C2F5E2C-DDE6-6748-ABF1-8E10CBF68CF9.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018B-UL2018_MiniAODv2-v1/out_694.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260000/3C982F8F-A3FB-C343-B77C-82EDE356B9C8.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018B-UL2018_MiniAODv2-v1/out_695.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260000/EF674E10-CE42-6F4A-98EA-BDC5CAE85FC5.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018B-UL2018_MiniAODv2-v1/out_698.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260000/52E79DA9-79AC-E846-B5D5-44B5150001BF.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018B-UL2018_MiniAODv2-v1/out_699.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260000/48A5EFBB-E501-1E45-B1B0-4753668E66ED.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018B-UL2018_MiniAODv2-v1/out_222.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260001/981FC325-0AA8-F942-AD90-EDCEA5D89FB9.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018B-UL2018_MiniAODv2-v1/out_542.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260001/C8823E8B-C825-1B4F-B12C-938E8A697941.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018B-UL2018_MiniAODv2-v1/out_543.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260001/147D1D73-2EA2-D04A-969E-A590292AF1FB.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018B-UL2018_MiniAODv2-v1/out_540.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260001/6F2D2232-A4B5-3845-A1AD-E0BAA51D1B70.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018B-UL2018_MiniAODv2-v1/out_541.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260001/21F026F7-E54E-DF4E-BA12-3995182DB48B.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018B-UL2018_MiniAODv2-v1/out_546.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260001/CDA2A1BC-4739-B94E-BD65-A5A26746DD95.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018B-UL2018_MiniAODv2-v1/out_547.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260001/45547020-4C07-5347-8E15-84B8A034AA0C.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018B-UL2018_MiniAODv2-v1/out_544.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260001/A6B0D607-65EA-B84F-A101-1C1988174C36.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018B-UL2018_MiniAODv2-v1/out_545.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260002/1D3C3A00-591D-9344-B035-65838B71B5AA.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018B-UL2018_MiniAODv2-v1/out_1018.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260001/653A63F7-8532-8C4D-BFE4-57C29B425228.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018B-UL2018_MiniAODv2-v1/out_548.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260001/BC656194-1B9D-514D-A629-F94E6F147174.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018B-UL2018_MiniAODv2-v1/out_549.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260000/78BC0B88-0656-2042-B9E1-BBDEA0B2FD97.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018B-UL2018_MiniAODv2-v1/out_827.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260002/4827D3D8-36D7-A841-B285-558758494D68.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018B-UL2018_MiniAODv2-v1/out_1019.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260000/2D31265C-5337-204D-BE53-FB84638578B8.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018B-UL2018_MiniAODv2-v1/out_995.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260001/360EF924-492B-AA43-89E7-1EE2CA3A1451.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018B-UL2018_MiniAODv2-v1/out_994.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260000/15E38569-3570-654F-B34A-38487F3BCECB.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018B-UL2018_MiniAODv2-v1/out_997.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260000/88E4C23A-AFAB-8A45-BBAB-AC4863007264.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018B-UL2018_MiniAODv2-v1/out_996.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260001/E394E905-FF77-6D45-8E2D-30FA31E1D998.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018B-UL2018_MiniAODv2-v1/out_991.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260001/10FC15C2-8CD3-7A43-A894-6CE4ECDA37C5.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018B-UL2018_MiniAODv2-v1/out_990.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260001/48617C33-AFC0-D843-AE16-CDF25DAC6E92.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018B-UL2018_MiniAODv2-v1/out_993.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260001/C74EDE1F-8410-4341-903D-1FA7BA350D67.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018B-UL2018_MiniAODv2-v1/out_992.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260000/38192917-4E1B-7E47-B2CC-4AEA5B287C3D.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018B-UL2018_MiniAODv2-v1/out_999.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260000/3539DD43-C1F9-B944-A194-5CB2BE3F18C2.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018B-UL2018_MiniAODv2-v1/out_998.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260001/16E945EE-4EA3-8943-B448-1E351A947FD7.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018B-UL2018_MiniAODv2-v1/out_120.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260001/5CE5B02B-3041-6344-A7E8-9C0D6760BF90.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018B-UL2018_MiniAODv2-v1/out_121.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260001/C1955034-D5EB-1440-A703-E35F94D31EE8.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018B-UL2018_MiniAODv2-v1/out_122.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260001/60191B79-12CB-B24D-B6ED-674101C7197B.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018B-UL2018_MiniAODv2-v1/out_123.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260001/B64BACCC-E201-AD47-B58D-A339B3C7F17F.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018B-UL2018_MiniAODv2-v1/out_124.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260001/4D805D32-D30E-AD4A-8198-21EC899AEF56.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018B-UL2018_MiniAODv2-v1/out_125.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260001/377C1793-5E00-FE40-BB54-B3332252CF86.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018B-UL2018_MiniAODv2-v1/out_126.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260000/3512F498-517F-7742-BAC7-F31F6B04AF73.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018B-UL2018_MiniAODv2-v1/out_127.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260000/4C80A5FF-DD9B-9646-A2B6-6D8F65F2C1EF.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018B-UL2018_MiniAODv2-v1/out_128.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260001/502982C1-25A3-624F-BF86-A8D6FB60E41C.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018B-UL2018_MiniAODv2-v1/out_2.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260000/9F20E15C-707E-8444-9531-12F37E7ABC01.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018B-UL2018_MiniAODv2-v1/out_219.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260000/7A556F4C-2A43-164C-AAE0-EC30FCC44F4B.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018B-UL2018_MiniAODv2-v1/out_176.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260000/4CAD62F3-1132-BD43-8138-41243C3FD092.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018B-UL2018_MiniAODv2-v1/out_214.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260000/B78661C0-3EE0-B54E-B8D8-C09C5FAFFA03.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018B-UL2018_MiniAODv2-v1/out_777.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260002/5D3A66B8-F8C9-684C-8733-03EC9AA61BC4.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018B-UL2018_MiniAODv2-v1/out_1128.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260002/154C5001-5BFA-8949-905F-DF269420F614.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018B-UL2018_MiniAODv2-v1/out_1010.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260000/0EE2AB5E-E492-2E4C-BE14-5062949C7951.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018B-UL2018_MiniAODv2-v1/out_563.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260000/519FEA8B-1F50-E341-8ADA-6085B4C6D226.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018B-UL2018_MiniAODv2-v1/out_1022.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260002/E9F928EA-C5FF-0D49-BA49-AF0D1F44D3B7.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018B-UL2018_MiniAODv2-v1/out_1011.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260001/F4DE87B5-C157-7343-B601-ADF5CE049E9A.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018B-UL2018_MiniAODv2-v1/out_928.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260001/60FA9184-D6ED-6345-A733-BC1ECF19B58B.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018B-UL2018_MiniAODv2-v1/out_929.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260000/D10617AE-C6E7-404B-980E-7233FF09FCEF.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018B-UL2018_MiniAODv2-v1/out_416.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260000/01CD6E06-A979-7448-AEAE-C2A1C862DE66.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018B-UL2018_MiniAODv2-v1/out_417.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260000/A9F9F6D9-D8B3-1746-823D-64EB6BE6707C.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018B-UL2018_MiniAODv2-v1/out_410.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260000/328058CF-FF81-1E40-8333-847F7F48E7F8.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018B-UL2018_MiniAODv2-v1/out_411.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260001/56B2BF77-B07C-8940-8BBB-7AD3CB24CD11.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018B-UL2018_MiniAODv2-v1/out_412.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260001/FB7A4F9A-8256-AF4E-BAAD-6F33362AA43E.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018B-UL2018_MiniAODv2-v1/out_413.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260000/EA9D016D-C25E-4540-9CF7-BCE902D50413.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018B-UL2018_MiniAODv2-v1/out_920.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260000/D238F9F7-813A-2C4C-BA42-A77FFBA72AB6.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018B-UL2018_MiniAODv2-v1/out_498.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260000/F051A01C-D8EB-AB4C-BC1C-8B5B8B6CC88A.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018B-UL2018_MiniAODv2-v1/out_922.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260001/F7EAA106-F828-1745-88CF-CD97496989EE.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018B-UL2018_MiniAODv2-v1/out_923.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260001/514A4AF1-D05C-1F43-8AC5-BD72613703A1.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018B-UL2018_MiniAODv2-v1/out_924.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260001/3218DB7E-A02D-F941-9DBE-CF3EAF373B1E.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018B-UL2018_MiniAODv2-v1/out_925.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260001/E1333EC5-B93D-B94E-A367-761199033954.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018B-UL2018_MiniAODv2-v1/out_926.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260001/DCEC46A0-AFE0-684B-B2C1-AEAD396E1051.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018B-UL2018_MiniAODv2-v1/out_927.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260001/38C65C45-E942-0245-89D8-9D32BD42BAA9.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018B-UL2018_MiniAODv2-v1/out_319.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260001/F324DA14-E8B2-524D-9321-10825F39D236.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018B-UL2018_MiniAODv2-v1/out_318.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260001/776CD485-16E8-0942-8A75-DE7CE8A75F1F.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018B-UL2018_MiniAODv2-v1/out_313.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260001/8DE39AB1-B049-3046-8142-656FC3800C98.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018B-UL2018_MiniAODv2-v1/out_312.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260001/563933BB-3EA7-AF43-B99B-BD1DDDB6E2F0.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018B-UL2018_MiniAODv2-v1/out_311.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260001/00C209A3-330F-6A46-8E95-4A6A658113A0.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018B-UL2018_MiniAODv2-v1/out_310.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260001/96FA09C4-1D23-6343-804D-E3E4F76CF155.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018B-UL2018_MiniAODv2-v1/out_317.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260001/47B17B73-C410-564B-8112-B41D09D68879.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018B-UL2018_MiniAODv2-v1/out_316.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260001/C033A289-2B77-F548-95F8-F46CCC1A49E7.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018B-UL2018_MiniAODv2-v1/out_315.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260002/E60BB1C7-B63E-1948-8133-9C117E115F67.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018B-UL2018_MiniAODv2-v1/out_1028.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260000/6B4078B2-A348-AC4D-B629-6AC0F0090570.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018B-UL2018_MiniAODv2-v1/out_921.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260000/64D32DC8-ABAF-AE43-9CC3-862D8FE51CFD.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018B-UL2018_MiniAODv2-v1/out_496.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260002/FEEE263A-5EA7-514A-B7C5-E648B98834BA.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018B-UL2018_MiniAODv2-v1/out_1134.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260000/9D339075-9495-5F45-9245-54E7B1310F13.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018B-UL2018_MiniAODv2-v1/out_832.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260000/105EF1C3-B1AD-F44C-9D95-BB4B22CD84CF.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018B-UL2018_MiniAODv2-v1/out_833.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260000/0C34A9A9-2BC7-EA41-BB54-FA4B0C84E5CE.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018B-UL2018_MiniAODv2-v1/out_830.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260000/495C4E2D-4763-7845-BCF1-CBE511EBB275.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018B-UL2018_MiniAODv2-v1/out_831.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260000/C93A5DC3-88ED-8D47-A192-D681C72AE968.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018B-UL2018_MiniAODv2-v1/out_836.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260000/5693186D-94DC-B643-94BA-DA9B0FBD24D0.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018B-UL2018_MiniAODv2-v1/out_837.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260000/8BA5C155-5EE3-274F-88BD-D8A5A941C98A.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018B-UL2018_MiniAODv2-v1/out_834.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260000/8B7B6C02-B795-C544-8B4F-B0DB65A33CD4.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018B-UL2018_MiniAODv2-v1/out_835.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260000/DEC06465-9371-5045-8935-6860462D7C54.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018B-UL2018_MiniAODv2-v1/out_838.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260000/83CC54A4-EC36-E745-8FE7-7AF4168759E7.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018B-UL2018_MiniAODv2-v1/out_839.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260001/42238FFB-60A6-1648-AE9E-60B119D88506.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018B-UL2018_MiniAODv2-v1/out_3.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260001/C9858836-20FC-8747-A154-FA42A62B9FF0.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018B-UL2018_MiniAODv2-v1/out_725.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260000/2041C1AE-58C4-F145-95B6-70BEC8AFC82B.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018B-UL2018_MiniAODv2-v1/out_368.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260000/4B2697EC-1580-304D-81AD-F6E43418E282.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018B-UL2018_MiniAODv2-v1/out_369.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260000/31FF559E-B79D-524F-A7C4-02E6BC8B5656.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018B-UL2018_MiniAODv2-v1/out_366.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260000/9C43DFFB-5D9C-144D-8FAD-6BEDB62FE439.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018B-UL2018_MiniAODv2-v1/out_367.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260000/5CEF9C1D-9D44-6742-9962-65321B199737.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018B-UL2018_MiniAODv2-v1/out_364.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260000/D05627BC-0553-B543-91F3-3987E492F7AA.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018B-UL2018_MiniAODv2-v1/out_365.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260000/1EC4E75A-BEC5-0049-8216-A04A8F9B2FD6.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018B-UL2018_MiniAODv2-v1/out_362.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260000/94305BBF-ABBF-AD44-BD08-0B246D98679C.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018B-UL2018_MiniAODv2-v1/out_363.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260000/6416209E-977A-7340-B501-4D13BBCC30F0.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018B-UL2018_MiniAODv2-v1/out_360.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260000/CBB1C228-330B-5247-8039-9F56AE77B15A.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018B-UL2018_MiniAODv2-v1/out_361.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260000/9EBCDA02-7243-9444-A923-EC25DCAF367B.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018B-UL2018_MiniAODv2-v1/out_218.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260000/54477270-E0A1-774E-B390-A400F3C8DC7D.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018B-UL2018_MiniAODv2-v1/out_380.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260001/7F7394A6-6AEE-514E-B8D4-43818C30D578.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018B-UL2018_MiniAODv2-v1/out_861.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260000/4A3BB481-FCA2-1E4F-9DE5-82382E7BFA92.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018B-UL2018_MiniAODv2-v1/out_382.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260001/25024B2E-F495-5A43-9CC4-ABF47D7F839D.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018B-UL2018_MiniAODv2-v1/out_383.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260001/A68C341F-271D-5546-BD6E-E21B25D3713B.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018B-UL2018_MiniAODv2-v1/out_384.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260001/8E083168-837D-4A41-AFB1-3CB25EED00CF.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018B-UL2018_MiniAODv2-v1/out_385.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260001/CC80E32F-089E-134E-A600-753D26DE1808.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018B-UL2018_MiniAODv2-v1/out_386.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260001/7D34EA0F-D47D-534F-A7C0-CD9E188A6D6C.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018B-UL2018_MiniAODv2-v1/out_387.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260001/6DDF7999-1BF4-4240-8B12-1996110CC8EE.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018B-UL2018_MiniAODv2-v1/out_388.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260001/7030363F-0761-E547-A274-BAE65CA7D5F4.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018B-UL2018_MiniAODv2-v1/out_389.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260000/AB7A5DB1-70CD-494B-B359-275C9FEB9F1D.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018B-UL2018_MiniAODv2-v1/out_784.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260000/81277548-0BF2-7149-8609-AAD7F84D8897.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018B-UL2018_MiniAODv2-v1/out_785.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260000/6946B15F-2FFA-5D4D-8595-C6BBEA920814.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018B-UL2018_MiniAODv2-v1/out_786.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260001/0DF7CEED-64BC-7F4E-A196-83B641796D99.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018B-UL2018_MiniAODv2-v1/out_787.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260000/31007AF8-2202-3249-B994-7CD880D80B83.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018B-UL2018_MiniAODv2-v1/out_780.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260000/5426E65E-06FD-0740-B73F-4FEAD1C90EF3.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018B-UL2018_MiniAODv2-v1/out_781.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260000/E9F81A61-D98B-2A47-9396-2E68A2160FDC.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018B-UL2018_MiniAODv2-v1/out_782.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260000/C5896572-CFEB-8146-9358-4C84899C8A8E.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018B-UL2018_MiniAODv2-v1/out_381.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260001/BBE5F05E-2DFC-4D46-85EF-E9FFBC0D8922.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018B-UL2018_MiniAODv2-v1/out_788.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260001/B628449D-51D2-E44F-A3A1-AEE0EA4684C4.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018B-UL2018_MiniAODv2-v1/out_789.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260002/ED6B268A-4F4F-1F4A-913B-32948CFF2FCD.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018B-UL2018_MiniAODv2-v1/out_1174.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260001/578426DC-231B-7546-945F-0383C2114321.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018B-UL2018_MiniAODv2-v1/out_860.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260001/8E7F3EC8-B91F-5343-B797-0703CA1C5E2D.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018B-UL2018_MiniAODv2-v1/out_630.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260000/0DDEDDE8-D077-8249-A267-28DB7822D629.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018B-UL2018_MiniAODv2-v1/out_114.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260002/5B529143-0D74-2E43-A6F1-4CE2D52E6CDA.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018B-UL2018_MiniAODv2-v1/out_1223.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260002/F2E441A0-64C2-C142-BF35-4854CC46899B.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018B-UL2018_MiniAODv2-v1/out_1164.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260000/826ED62B-1E58-E24D-8B7D-A9A72DA3B048.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018B-UL2018_MiniAODv2-v1/out_579.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260000/B2F9AAA8-1497-6F44-AF9B-60034DEA4D2D.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018B-UL2018_MiniAODv2-v1/out_578.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260002/6372ABF5-CFF3-E946-BFB6-A3642E82611F.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018B-UL2018_MiniAODv2-v1/out_1165.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260000/6430D842-DCA2-AA45-A858-EA2DC0DA3C73.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018B-UL2018_MiniAODv2-v1/out_573.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260000/1071C7EA-3AE5-E047-A249-9FF7D63DDE46.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018B-UL2018_MiniAODv2-v1/out_572.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260000/9655EC7E-A177-734A-B365-7FF7EAA2DECC.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018B-UL2018_MiniAODv2-v1/out_571.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260000/803BFD06-9627-6641-AD2B-9D92F8451B93.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018B-UL2018_MiniAODv2-v1/out_570.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260000/0052A064-DEF5-B347-A73F-CF413FCDA86E.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018B-UL2018_MiniAODv2-v1/out_577.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260000/2B0E2F2C-E4B9-E846-A752-CED51FDEE8BF.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018B-UL2018_MiniAODv2-v1/out_576.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260000/5EE04854-B1C5-2748-BC37-4AF58F9A8B48.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018B-UL2018_MiniAODv2-v1/out_575.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260000/B208729C-91A3-3849-B072-55F32F0EF0D9.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018B-UL2018_MiniAODv2-v1/out_574.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260000/F380661A-9A85-E948-BB02-8C5A45988379.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018B-UL2018_MiniAODv2-v1/out_60.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260000/63861130-3F65-E844-B3C3-5054E808E8D3.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018B-UL2018_MiniAODv2-v1/out_61.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260000/F59FFDBF-F0B4-7848-B336-20D48DCF2AE3.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018B-UL2018_MiniAODv2-v1/out_62.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260001/50060224-26BD-794E-89DB-13BC224741B0.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018B-UL2018_MiniAODv2-v1/out_259.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260000/482C6306-FB7B-5E44-B504-43B558EE139B.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018B-UL2018_MiniAODv2-v1/out_64.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260000/CA76AD16-DE2C-D94D-AD76-F53AB34542E5.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018B-UL2018_MiniAODv2-v1/out_65.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260000/88E07A0B-8D48-B044-8621-8012A4576FE4.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018B-UL2018_MiniAODv2-v1/out_66.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260000/ACFA03EF-95E9-3444-BC7D-827555C6DFC0.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018B-UL2018_MiniAODv2-v1/out_67.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260000/B0B431B4-5ED7-8641-B5CF-4C9E5A7D625D.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018B-UL2018_MiniAODv2-v1/out_68.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260001/B0B6C7D0-81F1-BD4A-A5A3-E2EB16AF67FB.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018B-UL2018_MiniAODv2-v1/out_253.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260001/F287843C-E4AF-CE4B-A2EF-B68B5BFD4802.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018B-UL2018_MiniAODv2-v1/out_250.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260001/9B708FE7-027D-3749-9A1C-DDA93FA9D391.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018B-UL2018_MiniAODv2-v1/out_251.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260001/6A8E66B1-17DC-EF49-949C-D6AEB14921CB.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018B-UL2018_MiniAODv2-v1/out_256.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260001/F97FA92F-F05F-DC4E-80BA-C99B53A4FDF0.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018B-UL2018_MiniAODv2-v1/out_257.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260001/463D86C5-150C-F74E-ABBB-96E5D71E9C68.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018B-UL2018_MiniAODv2-v1/out_254.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260001/BEE291C0-62F1-6347-8156-E6A6D321DB80.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018B-UL2018_MiniAODv2-v1/out_255.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260000/59DA7AC7-A256-AE46-B56A-6B78C5DA694C.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018B-UL2018_MiniAODv2-v1/out_499.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260001/F308A333-6BC6-AB44-A580-29DD1A6697E2.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018B-UL2018_MiniAODv2-v1/out_603.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260002/963275EB-1DD9-834A-A7DF-3629428F73A3.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018B-UL2018_MiniAODv2-v1/out_1172.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260000/163D7FD6-32D2-4441-A597-5096361F2CE8.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018B-UL2018_MiniAODv2-v1/out_156.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260001/95592CDC-458B-7E4A-B831-37865FE1A9E5.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018B-UL2018_MiniAODv2-v1/out_939.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260001/452306CE-99F0-A94F-9CBA-5DDBB0DE7ABF.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018B-UL2018_MiniAODv2-v1/out_731.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260001/D4C04BF4-6B85-6044-8605-36B39889F429.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018B-UL2018_MiniAODv2-v1/out_730.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260001/9F5AA8BB-77FC-0D4C-88E9-7123CFC0CC46.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018B-UL2018_MiniAODv2-v1/out_733.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260001/22530797-8040-3B40-919E-3280CAC72E3B.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018B-UL2018_MiniAODv2-v1/out_938.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260001/5B1165C2-BCC3-9042-B417-4457050C84BB.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018B-UL2018_MiniAODv2-v1/out_735.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260001/C9223B70-3912-5C4A-A72A-DC2768E19096.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018B-UL2018_MiniAODv2-v1/out_734.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260000/05D5B083-C3A1-D64F-B9DE-E8FD1A6983E7.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018B-UL2018_MiniAODv2-v1/out_508.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260001/6E588D5D-FD22-F944-8C3E-9AD4F2656BDC.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018B-UL2018_MiniAODv2-v1/out_736.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260002/1BABAF4C-5A21-3B42-BF60-FEB1EB1B12BA.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018B-UL2018_MiniAODv2-v1/out_1032.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260001/4E77822C-5947-BB4F-AEAB-DB32C7401DBE.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018B-UL2018_MiniAODv2-v1/out_738.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260002/750ED02E-C0FB-E443-84FA-58D3E9AD7077.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018B-UL2018_MiniAODv2-v1/out_1030.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260000/4D811DEC-7126-AA4A-B082-AB4CEF9181D5.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018B-UL2018_MiniAODv2-v1/out_505.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260000/60A74473-99D7-6C4E-87DC-797FA6902AC7.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018B-UL2018_MiniAODv2-v1/out_502.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260000/84881E77-CDC0-B145-9060-94C18E53C1ED.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018B-UL2018_MiniAODv2-v1/out_503.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260000/5205595E-0E79-5649-927B-0E8445FE97BC.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018B-UL2018_MiniAODv2-v1/out_500.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260000/31AAF029-F1BB-E646-8CF9-C6D204D6BAA1.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018B-UL2018_MiniAODv2-v1/out_501.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260002/6849C81F-AD1E-5B4F-B2A0-FBAAEDD0E1EF.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018B-UL2018_MiniAODv2-v1/out_1212.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260002/FC2B0079-7B94-E74D-BAB7-06335C66FBF8.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018B-UL2018_MiniAODv2-v1/out_1213.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260000/336E3566-9790-3347-9686-C345388889DF.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018B-UL2018_MiniAODv2-v1/out_632.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260000/FFA6E583-5B24-B34B-A5C2-4238DE01C940.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018B-UL2018_MiniAODv2-v1/out_633.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260000/34D257A0-2D1D-1744-A98B-E385B3D8417A.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018B-UL2018_MiniAODv2-v1/out_1216.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260000/1CB68864-2154-CE4D-836B-00BEFDD16776.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018B-UL2018_MiniAODv2-v1/out_468.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260000/2063478B-B7AA-C643-9C1D-6335B3CC9B0C.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018B-UL2018_MiniAODv2-v1/out_636.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260000/08694B22-235F-4046-BDD4-1F4BFDCA82A0.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018B-UL2018_MiniAODv2-v1/out_637.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260001/1ADE1117-4736-B142-A9DC-779A14185EB0.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018B-UL2018_MiniAODv2-v1/out_465.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260001/16CE95BF-3337-FA45-9BDD-AE0705338696.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018B-UL2018_MiniAODv2-v1/out_464.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260000/28AF87C7-7BDD-BD48-9E5D-552ED52804C6.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018B-UL2018_MiniAODv2-v1/out_1218.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260000/570CDC7E-B38B-AC46-8CAA-7E6C2D1DE1F3.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018B-UL2018_MiniAODv2-v1/out_466.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260001/E792C88D-F46F-4243-8446-81A581C99B6E.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018B-UL2018_MiniAODv2-v1/out_461.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260000/C36442C7-B3AB-5F41-81AB-FF78DD80E83E.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018B-UL2018_MiniAODv2-v1/out_900.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260001/25C131BB-8DB9-0E4C-BCD1-4932E64B7BBE.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018B-UL2018_MiniAODv2-v1/out_463.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260001/CDB125B0-4631-4B4E-904A-2CF6FA44534B.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018B-UL2018_MiniAODv2-v1/out_462.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260000/8B37E3E6-ECC6-A041-B889-2318B864E333.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018B-UL2018_MiniAODv2-v1/out_281.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260000/9AB62136-C56A-354B-84E5-0826E8B3EBA0.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018B-UL2018_MiniAODv2-v1/out_901.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260001/68F0810C-C0E8-3D46-8B8E-1EBC0FBDBB06.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018B-UL2018_MiniAODv2-v1/out_168.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260001/8E773706-5605-EA42-B866-4D676A267A62.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018B-UL2018_MiniAODv2-v1/out_169.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260001/5C90B818-9026-624B-A96E-2A06C3AE44DF.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018B-UL2018_MiniAODv2-v1/out_164.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260001/99ABAD46-EFAE-434C-B34F-FA611946DBAC.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018B-UL2018_MiniAODv2-v1/out_165.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260001/76320626-40D9-7047-8E40-05A3145DD5C9.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018B-UL2018_MiniAODv2-v1/out_166.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260001/FAD47EFF-2AAD-924E-A767-C6542EC6C604.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018B-UL2018_MiniAODv2-v1/out_167.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260000/8384510B-C188-F04D-A8F8-B3251984035F.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018B-UL2018_MiniAODv2-v1/out_160.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260000/CD2E8201-F9D3-6640-91A6-ED132F887626.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018B-UL2018_MiniAODv2-v1/out_161.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260002/69B77FA0-32B9-8547-A25E-6C18981A634B.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018B-UL2018_MiniAODv2-v1/out_1016.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260001/BC70D7C6-A85B-554D-AEB0-D3461BFEB9E2.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018B-UL2018_MiniAODv2-v1/out_163.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260000/4EAE8F3B-D938-6E48-B97C-E737A9BCB8E0.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018B-UL2018_MiniAODv2-v1/out_964.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260000/4A2536AD-BA98-EB46-99DB-3C47805B289F.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018B-UL2018_MiniAODv2-v1/out_965.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260000/780C16D5-2840-604D-B416-9DE55F76E23C.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018B-UL2018_MiniAODv2-v1/out_966.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260000/96A9223A-9CF1-BA46-9C6C-850866C89485.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018B-UL2018_MiniAODv2-v1/out_967.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260001/2E4FBC08-5D90-F64C-AC4D-870685C4FA6A.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018B-UL2018_MiniAODv2-v1/out_960.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260001/273B6DEB-066B-3C49-810F-61F14726E4B8.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018B-UL2018_MiniAODv2-v1/out_961.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260001/407DA819-04A1-5143-B7EA-1DF7AE309EA6.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018B-UL2018_MiniAODv2-v1/out_962.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260001/364EC0CB-E7FA-674A-9CE1-98F1C854E6BD.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018B-UL2018_MiniAODv2-v1/out_963.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260002/E0338E4B-8F1A-A64D-A645-D5F9B0C22290.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018B-UL2018_MiniAODv2-v1/out_1129.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260002/C2229E3B-0D59-DB43-BF99-70ED656DF812.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018B-UL2018_MiniAODv2-v1/out_1039.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260000/E7F4C66B-2A1B-B24A-869D-A22E82231EB3.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018B-UL2018_MiniAODv2-v1/out_783.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260000/53771E94-6E76-FC4E-9DEC-B139D57E0C7A.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018B-UL2018_MiniAODv2-v1/out_968.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260000/30A3BBB9-F92C-A440-94D0-A580FE9D15F8.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018B-UL2018_MiniAODv2-v1/out_969.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260001/D196D425-4A8B-2C47-A46B-742AAC0288AC.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018B-UL2018_MiniAODv2-v1/out_936.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260002/DE4BFF8B-A2D6-394B-AAAA-5FA7AF289D32.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018B-UL2018_MiniAODv2-v1/out_1106.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260002/27925995-7BB6-9C4D-919B-8B14EA178B31.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018B-UL2018_MiniAODv2-v1/out_1107.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260002/15BA6C2E-F8D7-7547-862C-381590263663.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018B-UL2018_MiniAODv2-v1/out_1104.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260002/EEB7A567-B954-484A-8EDD-0DA0CAB3DA92.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018B-UL2018_MiniAODv2-v1/out_1105.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260002/C2BBBDE0-EEE2-1843-BE71-67D3F57AACD7.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018B-UL2018_MiniAODv2-v1/out_1102.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260000/65A69C9D-132B-D84F-8CEC-929B5CEDD41C.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018B-UL2018_MiniAODv2-v1/out_1103.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260002/04FDABE4-3C5B-8743-B9CC-60A97E85A1CF.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018B-UL2018_MiniAODv2-v1/out_1100.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260002/01F2C780-0CAA-664B-A632-A3064DE3AD76.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018B-UL2018_MiniAODv2-v1/out_1101.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260001/D5790002-9AE6-0D49-B45E-BB0152E2C38E.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018B-UL2018_MiniAODv2-v1/out_934.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260000/857A445C-8BC4-9344-A6E3-1DAE1D4718CA.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018B-UL2018_MiniAODv2-v1/out_908.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260000/E34621F9-A180-F74E-97A8-5B54136643AE.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018B-UL2018_MiniAODv2-v1/out_1108.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260002/B3AC4E61-493A-9C49-9286-5379EA1D33E8.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018B-UL2018_MiniAODv2-v1/out_1109.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260001/180DACD1-8CD1-954D-ABAE-C954A92232FD.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018B-UL2018_MiniAODv2-v1/out_670.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260000/4A47E58B-9057-694F-9F90-4F27BBD4471B.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018B-UL2018_MiniAODv2-v1/out_909.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260000/78B1EE60-E6A5-8343-AAD6-D596CDDDFA1A.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018B-UL2018_MiniAODv2-v1/out_635.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260001/D0538E7D-F8A3-3E40-9CCD-AE5350E9C169.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018B-UL2018_MiniAODv2-v1/out_722.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260000/9A030E9B-0CD5-7B42-8CE2-EB4FF87204EC.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018B-UL2018_MiniAODv2-v1/out_584.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260000/4AD9B758-38B9-4540-A472-BE79A461EDB2.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018B-UL2018_MiniAODv2-v1/out_678.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260001/760E6B67-2224-354C-A904-AE20D4256FE1.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018B-UL2018_MiniAODv2-v1/out_609.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260002/D7B6B285-10FA-DE41-B90F-4A24CA991B7C.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018B-UL2018_MiniAODv2-v1/out_1158.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260000/44F16520-38D7-1E4F-8B5F-C93047B10B40.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018B-UL2018_MiniAODv2-v1/out_426.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260001/D8CF7CA2-1708-D042-84C0-A69D97253F82.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018B-UL2018_MiniAODv2-v1/out_878.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260001/7A4A57ED-548E-B34D-A683-EA508FA4B5E1.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018B-UL2018_MiniAODv2-v1/out_879.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260001/6648C652-0917-3341-B81C-E4E84474268E.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018B-UL2018_MiniAODv2-v1/out_876.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260001/39544086-FEBF-7548-84CB-1ECA30CA24F3.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018B-UL2018_MiniAODv2-v1/out_877.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260001/F6D99A9C-ABA6-054B-993D-A5F1312C3A63.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018B-UL2018_MiniAODv2-v1/out_874.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260001/911E6C4B-BEE4-D44F-A044-8F1211324D08.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018B-UL2018_MiniAODv2-v1/out_875.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260001/422EFAFA-EB2B-3549-ACA5-50FAE31AD630.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018B-UL2018_MiniAODv2-v1/out_872.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260001/F2AAF54C-BC28-D04D-A8FB-79149D093CCB.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018B-UL2018_MiniAODv2-v1/out_873.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260001/8E06E0AA-EC66-6F40-9AD3-F389BD72E68D.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018B-UL2018_MiniAODv2-v1/out_870.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260001/15CF3978-DF07-0644-ACB8-60D07FA3CCB1.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018B-UL2018_MiniAODv2-v1/out_871.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260001/79D376DA-7020-1F44-B2D4-A0791D712801.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018B-UL2018_MiniAODv2-v1/out_674.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260001/6EB4F9B2-4F56-6143-93D6-A615B8B07962.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018B-UL2018_MiniAODv2-v1/out_675.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260001/E98BC7B1-4772-9A45-97F8-6C7C25E2CBA0.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018B-UL2018_MiniAODv2-v1/out_9.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260000/117C7CED-6B63-1740-9B41-33362A861933.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018B-UL2018_MiniAODv2-v1/out_676.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260000/857DF82C-EED2-9C4B-8A56-1377ACE41089.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018B-UL2018_MiniAODv2-v1/out_890.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260000/39C064E2-5445-7F4E-92F7-24655B1E29EF.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018B-UL2018_MiniAODv2-v1/out_891.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260000/022E0AF6-828F-8B45-B52E-C2C10DA5FE73.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018B-UL2018_MiniAODv2-v1/out_892.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260000/DA48638E-2C6E-BE4F-925C-4D7F7EDBF086.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018B-UL2018_MiniAODv2-v1/out_893.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260000/3EB48644-A6F3-2A48-BECF-F0CE6ED716C8.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018B-UL2018_MiniAODv2-v1/out_894.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260000/33AC4454-CD6B-B14C-8DD0-DD9E451F8FCB.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018B-UL2018_MiniAODv2-v1/out_647.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260000/8A65A366-2DFE-6E4C-99FE-C66B8EF013FB.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018B-UL2018_MiniAODv2-v1/out_896.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260000/D54D98BA-F14E-1046-96CD-29E34228AC19.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018B-UL2018_MiniAODv2-v1/out_897.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260000/E5B89278-DEF5-4F4F-B07C-E53E9D07E853.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018B-UL2018_MiniAODv2-v1/out_898.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260000/4D105020-A833-B54D-81B4-BC6644BEE9E2.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018B-UL2018_MiniAODv2-v1/out_899.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260000/48AC2769-BDE4-4F43-821D-CFABE9FF363A.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018B-UL2018_MiniAODv2-v1/out_1087.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260000/2A949DDE-9848-844C-BC78-33FD680329D7.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018B-UL2018_MiniAODv2-v1/out_646.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260002/6B5A0230-69D4-194E-B010-B32D756AE493.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018B-UL2018_MiniAODv2-v1/out_1099.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260002/52BEB33D-A08B-E440-B0C6-98386FCA4381.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018B-UL2018_MiniAODv2-v1/out_1138.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260002/4A360FB8-10D5-D84C-91BD-B87A08E705F9.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018B-UL2018_MiniAODv2-v1/out_1199.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260001/FCEFF70E-AA04-0240-AFFC-8E67C86839D0.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018B-UL2018_MiniAODv2-v1/out_672.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260000/CE3D255D-192A-F842-AAFE-F528602F541F.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018B-UL2018_MiniAODv2-v1/out_648.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260001/6B9DBCD2-37C7-304E-BC4B-47244C340773.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018B-UL2018_MiniAODv2-v1/out_673.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260000/543DEA34-AE09-0B41-B0DD-C0991742E862.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018B-UL2018_MiniAODv2-v1/out_1036.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260000/9E78D24C-7ACC-C74F-B05D-93C5EA74B7F6.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018B-UL2018_MiniAODv2-v1/out_638.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260002/9082C8BC-9C32-B146-8C4D-72CFE15DD53A.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018B-UL2018_MiniAODv2-v1/out_1173.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260000/DBA9BBD8-CE95-AA40-97C0-17C5AAED2AE8.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018B-UL2018_MiniAODv2-v1/out_1086.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260000/85AE04CC-1573-B446-80B3-20FF9FC3E879.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018B-UL2018_MiniAODv2-v1/out_357.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260002/28722510-550D-404A-9EA4-4E797CF41BAC.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018B-UL2018_MiniAODv2-v1/out_1198.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260000/86EC7E6B-F0BF-E047-A841-48E9FB285800.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018B-UL2018_MiniAODv2-v1/out_809.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260000/D3D79BFC-E849-9D48-AAE4-7D649EB561EB.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018B-UL2018_MiniAODv2-v1/out_808.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260000/A65333AE-92B6-9346-8DA1-6695CDE9E532.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018B-UL2018_MiniAODv2-v1/out_353.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260000/D670E77C-D28C-D249-BB2C-A7071B94699C.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018B-UL2018_MiniAODv2-v1/out_352.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260000/57566F3B-23BA-D648-ABFD-911EEBB02ECF.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018B-UL2018_MiniAODv2-v1/out_351.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260000/83E2DD90-F86F-D24D-97B4-0A6B12E4CB88.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018B-UL2018_MiniAODv2-v1/out_350.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260001/DB38D112-F944-834B-B3E4-42D9237D8FE4.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018B-UL2018_MiniAODv2-v1/out_803.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260001/01FFA3BB-66C4-8548-8282-96ED18853FE7.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018B-UL2018_MiniAODv2-v1/out_802.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260001/2C0EC3A1-91B8-AB42-ABD1-C124F2E44DBF.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018B-UL2018_MiniAODv2-v1/out_801.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260001/6512DAEF-F59B-7946-AE2B-31B505A7F1A4.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018B-UL2018_MiniAODv2-v1/out_800.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260001/52F10F89-77CD-3546-B236-1F8A1317CEF3.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018B-UL2018_MiniAODv2-v1/out_807.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260001/8E2B99C4-97C9-C343-8322-7688E9758942.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018B-UL2018_MiniAODv2-v1/out_806.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260001/932035ED-6867-AE46-9911-292E729612A4.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018B-UL2018_MiniAODv2-v1/out_805.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260001/924A14DB-2472-834A-90E0-26620319C3E5.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018B-UL2018_MiniAODv2-v1/out_804.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260000/1CD2BA15-5B50-6E4A-9DA1-FAB886FFAF4B.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018B-UL2018_MiniAODv2-v1/out_216.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260000/4E24B6D3-E9DA-3E4E-AD7A-5B7ADA825371.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018B-UL2018_MiniAODv2-v1/out_217.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260000/C383CFC0-46E3-BD44-9014-6018B904AC11.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018B-UL2018_MiniAODv2-v1/out_768.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260000/33A52278-377D-0242-94DE-87F23453965F.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018B-UL2018_MiniAODv2-v1/out_769.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260000/96264FC7-479B-0941-AC6F-932D9B6B01A1.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018B-UL2018_MiniAODv2-v1/out_212.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260000/0E9D5B7C-1486-2142-8BC3-9A026DB6A98B.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018B-UL2018_MiniAODv2-v1/out_213.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260000/E2FC8AB3-1452-4640-9DC1-EDB4A3910A2F.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018B-UL2018_MiniAODv2-v1/out_210.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260000/8224E41A-FC43-654B-A061-52ABFD98DE92.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018B-UL2018_MiniAODv2-v1/out_211.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260000/4845E10D-82F0-0245-9987-8DA57BC781AB.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018B-UL2018_MiniAODv2-v1/out_762.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260000/4FFE4C96-62AE-9B47-803E-145555310293.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018B-UL2018_MiniAODv2-v1/out_763.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260000/2AC288CD-A08D-8647-A3A5-E1DF4FE4BA26.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018B-UL2018_MiniAODv2-v1/out_760.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260000/71D109EC-A71B-CE4D-A3D8-BB6A77B77974.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018B-UL2018_MiniAODv2-v1/out_761.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260000/71880C7B-87EB-A549-8033-FAE0E26F091B.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018B-UL2018_MiniAODv2-v1/out_766.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260000/E95ADA3F-5C0E-2542-9888-79FEE71B9D20.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018B-UL2018_MiniAODv2-v1/out_767.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260000/0AB8E057-57E7-E746-92AA-20F53AE6DEF6.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018B-UL2018_MiniAODv2-v1/out_764.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260000/3B47AACA-E637-F24C-8C30-1D9DB046D180.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018B-UL2018_MiniAODv2-v1/out_765.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260000/AEDAF89B-F374-7A46-A76E-95950BB48EEE.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018B-UL2018_MiniAODv2-v1/out_40.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260000/569D1E8C-9B7F-5042-A93B-1BA451DB8D8D.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018B-UL2018_MiniAODv2-v1/out_41.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260002/762EF53D-54B1-644A-A231-A51D3408BF8B.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018B-UL2018_MiniAODv2-v1/out_1033.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260002/478CC7B9-DD3D-534A-A67B-4849E7D541DA.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018B-UL2018_MiniAODv2-v1/out_1169.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260002/86D64E87-B70C-044C-9142-DB0380FB43F4.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018B-UL2018_MiniAODv2-v1/out_1119.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260002/F17FB77D-F284-7148-837E-51058D003977.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018B-UL2018_MiniAODv2-v1/out_1078.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260002/B34C35E0-D1C7-C447-B0DA-E8FFBF332E7C.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018B-UL2018_MiniAODv2-v1/out_1079.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260001/16DDC64D-5EBA-804C-B971-2EEC5BAC3D34.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018B-UL2018_MiniAODv2-v1/out_1076.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260001/E9D2055F-803C-2D4C-89DE-F0CEF791655B.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018B-UL2018_MiniAODv2-v1/out_1077.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260000/79A94235-BF89-E546-9EC9-6EED94C87703.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018B-UL2018_MiniAODv2-v1/out_1074.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260001/21C64454-FE94-BD44-9F24-037B3B8FD0A1.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018B-UL2018_MiniAODv2-v1/out_1075.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260002/95B7CA07-D7FA-AB42-942B-FB2B5395EF9B.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018B-UL2018_MiniAODv2-v1/out_1072.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260002/24D4420C-DC81-5A47-A266-89C3EE63DCF5.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018B-UL2018_MiniAODv2-v1/out_1073.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260002/F44095FF-0C14-704A-89FA-24A7D468FD87.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018B-UL2018_MiniAODv2-v1/out_1070.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260002/D7A2D44F-848E-6345-84BE-DD9106641E83.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018B-UL2018_MiniAODv2-v1/out_1071.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260000/2AA50832-2236-BF4B-B269-8E62C784F07E.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018B-UL2018_MiniAODv2-v1/out_289.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260000/17B971A3-122F-4241-AE3D-950002EFF62B.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018B-UL2018_MiniAODv2-v1/out_288.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260002/4C5A1678-B4BE-4B44-AB98-9B003EAE5B87.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018B-UL2018_MiniAODv2-v1/out_1003.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260001/420B61DE-9C08-2144-BC78-C8B45CB9221B.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018B-UL2018_MiniAODv2-v1/out_4.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260002/E6AC6917-12D1-3C4D-91D1-633A2ED6A5DA.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018B-UL2018_MiniAODv2-v1/out_1179.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260000/240BF3A2-2716-2849-8F9F-1D38F2C6C46A.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018B-UL2018_MiniAODv2-v1/out_280.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260000/92AC5873-2004-F94E-BB97-9AD045EE9DAB.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018B-UL2018_MiniAODv2-v1/out_283.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260000/768E5BD8-983F-7C4A-B46A-684391BF9FBB.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018B-UL2018_MiniAODv2-v1/out_282.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260000/D68899C9-CFA8-E247-B7A1-629C879B0DE8.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018B-UL2018_MiniAODv2-v1/out_285.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260000/DB483C18-D247-2644-AC45-1DFFC99FCB00.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018B-UL2018_MiniAODv2-v1/out_284.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260000/B796CE0E-26D1-AD4D-9B32-DAFDCA1F0884.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018B-UL2018_MiniAODv2-v1/out_287.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260000/462E903F-E526-AC46-979B-007FE51B4670.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018B-UL2018_MiniAODv2-v1/out_286.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260000/34D0440C-C3CD-C74E-828E-5B985D23C387.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018B-UL2018_MiniAODv2-v1/out_1094.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260000/09A18A78-973D-F84A-ADD5-D3D1CAA56936.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018B-UL2018_MiniAODv2-v1/out_1095.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260000/51D45C22-E082-754F-8733-4CE66087EE8C.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018B-UL2018_MiniAODv2-v1/out_1096.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260000/B3D1024E-0922-4945-8A4D-94C5D09887D0.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018B-UL2018_MiniAODv2-v1/out_1097.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260001/E1CA8094-7C34-E94E-B23F-D2D55A650456.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018B-UL2018_MiniAODv2-v1/out_1090.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260001/E8129812-CC1B-EE4D-B97D-61F1198804DF.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018B-UL2018_MiniAODv2-v1/out_1091.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260002/658E3445-DE3C-5540-8327-416F9F963CBC.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018B-UL2018_MiniAODv2-v1/out_1092.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260002/547C957F-C388-E64B-A1B9-C8CB9E3B58CE.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018B-UL2018_MiniAODv2-v1/out_1093.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260002/BB75A55C-0F0D-CA46-97C9-ABA0CF355E11.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018B-UL2018_MiniAODv2-v1/out_1155.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260002/B2419E0A-11C0-A64D-83CA-102D02EC9DBC.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018B-UL2018_MiniAODv2-v1/out_1154.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260002/4891391E-C0D7-4547-AB5C-027EA92E3F49.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018B-UL2018_MiniAODv2-v1/out_1157.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260002/99E3EC3A-18C2-FC41-B333-3A8B7AFCE5DB.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018B-UL2018_MiniAODv2-v1/out_1156.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260000/32C92873-9924-204D-B8F3-27006FF64808.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018B-UL2018_MiniAODv2-v1/out_1098.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260002/E9328F90-7AB6-EE4A-AC70-4B2112ED2780.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018B-UL2018_MiniAODv2-v1/out_1150.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260002/76CACF3D-53A0-3044-9229-85BC31DCE784.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018B-UL2018_MiniAODv2-v1/out_1153.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260002/70CF8900-CEDC-6744-8A1B-C98893A5FE74.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018B-UL2018_MiniAODv2-v1/out_1152.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260001/57A5B39E-51AE-914B-98D9-4A8E5D2CE59A.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018B-UL2018_MiniAODv2-v1/out_263.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260001/71514E35-C808-E542-AC92-B264827501C7.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018B-UL2018_MiniAODv2-v1/out_262.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260001/0C47472E-6DC6-7649-9B53-970FB2485156.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018B-UL2018_MiniAODv2-v1/out_261.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260001/D9951429-F96E-1046-A794-3BB2088AD7D8.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018B-UL2018_MiniAODv2-v1/out_260.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260001/B17496B5-7CB8-3D48-B232-896782ED3798.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018B-UL2018_MiniAODv2-v1/out_267.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260001/E4E7CE01-CF8D-2946-8941-731D91802BD3.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018B-UL2018_MiniAODv2-v1/out_266.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260001/D99A60D5-4EA0-0D42-997D-C8B63801DC90.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018B-UL2018_MiniAODv2-v1/out_265.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260001/5CD044F8-2864-F542-9682-1EFBCA61FBDD.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018B-UL2018_MiniAODv2-v1/out_264.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260002/DD8F8FD7-EB44-DF48-9BC1-98BAA919DB9F.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018B-UL2018_MiniAODv2-v1/out_1121.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260000/AB661FED-B245-F145-9C74-73CA56FFE194.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018B-UL2018_MiniAODv2-v1/out_269.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260001/BD7B5315-1309-784A-B68B-4E150176B17A.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018B-UL2018_MiniAODv2-v1/out_268.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260001/F95B7EF2-6123-BD44-A305-5256241CE1E3.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018B-UL2018_MiniAODv2-v1/out_478.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260000/47E02891-88F8-B64B-93AD-1BEB7D489E21.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018B-UL2018_MiniAODv2-v1/out_59.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260000/0753F502-9547-CA42-AA1C-BB1F8801741D.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018B-UL2018_MiniAODv2-v1/out_58.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260000/91D7D318-2ECA-064C-B62E-D191C87E8511.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018B-UL2018_MiniAODv2-v1/out_1215.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260001/B105A464-2B99-C848-942D-80E503CFF383.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018B-UL2018_MiniAODv2-v1/out_55.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260001/7AFFC2E4-A0D5-5542-88C1-F5C8D2484D26.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018B-UL2018_MiniAODv2-v1/out_54.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260001/A37A17FE-90CD-1143-AED0-AA768296D36B.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018B-UL2018_MiniAODv2-v1/out_57.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260001/C56F3B24-7A0E-B54F-9958-C7013DF40401.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018B-UL2018_MiniAODv2-v1/out_56.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260001/D5AEFB8F-216B-F947-80AC-AF3B5C0FD5FE.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018B-UL2018_MiniAODv2-v1/out_51.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260001/54BA3A55-DA7E-AB43-A3D0-2A887541A166.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018B-UL2018_MiniAODv2-v1/out_258.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260001/72F3AC2A-0BF1-F845-9746-8181B3BC75C9.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018B-UL2018_MiniAODv2-v1/out_53.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260001/D6318F32-DE42-B546-9AC2-CA5476936641.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018B-UL2018_MiniAODv2-v1/out_52.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260001/F190568C-1EFC-A241-8241-26C8C7E90065.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018B-UL2018_MiniAODv2-v1/out_537.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260000/A1AD4075-5692-4949-B64D-CD9CD5477727.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018B-UL2018_MiniAODv2-v1/out_536.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260000/AC330765-0C0E-A043-887A-C7A4B67B30EB.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018B-UL2018_MiniAODv2-v1/out_535.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260000/291E24BD-46B5-5A49-93D0-85830ACA6F76.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018B-UL2018_MiniAODv2-v1/out_63.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260001/0EEF6E91-5D6D-E047-9086-15401CC83BB3.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018B-UL2018_MiniAODv2-v1/out_533.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260001/1F90D7E8-F8FD-6843-AB28-8C221E1335F5.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018B-UL2018_MiniAODv2-v1/out_532.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260001/FFE48EE9-4462-0B4A-9665-B9A7E300E39D.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018B-UL2018_MiniAODv2-v1/out_531.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260001/0BF0217A-AF82-6B44-8B2B-15DBA02B578B.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018B-UL2018_MiniAODv2-v1/out_530.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260001/AAE215DF-80CA-8941-BE11-CE59D9F14F62.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018B-UL2018_MiniAODv2-v1/out_598.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260000/B1851CE3-DAD1-BD47-94DD-3A40C2DE369A.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018B-UL2018_MiniAODv2-v1/out_152.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260000/ECA100C8-2D27-2D43-BA6F-8CCAC171E64C.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018B-UL2018_MiniAODv2-v1/out_539.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260000/6C857C63-4F26-1346-8F8E-55CA92114E1F.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018B-UL2018_MiniAODv2-v1/out_538.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260000/70C0701D-0122-3849-8963-DAB149AEA9EE.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018B-UL2018_MiniAODv2-v1/out_775.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260001/B70CB07F-381F-9243-A450-35650EA1D6E5.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018B-UL2018_MiniAODv2-v1/out_988.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260001/18C3A4A7-50BB-DB48-8A4E-B187699F4EBE.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018B-UL2018_MiniAODv2-v1/out_989.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260002/04CDA5AF-C206-024A-A15E-6F95A33932C0.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018B-UL2018_MiniAODv2-v1/out_1171.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260000/584CFFEA-04F6-9741-9877-9727469FD92F.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018B-UL2018_MiniAODv2-v1/out_774.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260000/60714C85-5464-D347-8FBE-2BED6DFED6DA.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018B-UL2018_MiniAODv2-v1/out_982.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260000/0BCB10BA-FE07-FB40-9536-9CD5AA740450.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018B-UL2018_MiniAODv2-v1/out_983.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260001/BDDD5EAA-EE32-384D-B165-181A1E4B3EE6.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018B-UL2018_MiniAODv2-v1/out_980.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260000/BA5360E0-044E-3A4C-B121-98446AFD7256.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018B-UL2018_MiniAODv2-v1/out_981.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260001/15CA2A8A-DF40-4841-9F24-A8CD676B1559.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018B-UL2018_MiniAODv2-v1/out_986.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260001/EA9C3ABC-93D9-CD4A-B835-3537F1F35C2F.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018B-UL2018_MiniAODv2-v1/out_987.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260000/8A27A3CF-0807-544F-807F-747E522B2935.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018B-UL2018_MiniAODv2-v1/out_984.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260001/2CB5C7FC-69DC-8E41-A0E1-CDE00222FC81.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018B-UL2018_MiniAODv2-v1/out_50.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260001/C514F862-D97A-E948-972C-B6EA686FA23A.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018B-UL2018_MiniAODv2-v1/out_115.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260001/8E7AF09B-237B-0541-9BFC-B26B4D0F9E67.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018B-UL2018_MiniAODv2-v1/out_252.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260001/A0F27C80-3D0E-7E4C-88BA-814715864A85.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018B-UL2018_MiniAODv2-v1/out_117.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260000/D50EDB46-9CD6-1748-838B-7AA283D3E56F.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018B-UL2018_MiniAODv2-v1/out_776.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260001/B27B2440-8C18-6146-9CA2-CE608195855C.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018B-UL2018_MiniAODv2-v1/out_111.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260001/01928D45-409B-FB4D-B6C0-89E1A6202F49.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018B-UL2018_MiniAODv2-v1/out_110.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260000/E7F16F44-3BDA-1A4C-81F6-09922B6FD8BE.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018B-UL2018_MiniAODv2-v1/out_113.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260000/2B3345D6-A4A5-DF4E-A311-2B4FDBACA16F.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018B-UL2018_MiniAODv2-v1/out_69.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260000/BAFE9031-E5C4-8B41-987D-BEC579CDD350.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018B-UL2018_MiniAODv2-v1/out_677.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260000/11414004-44A1-6449-8C28-3897751DC9CC.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018B-UL2018_MiniAODv2-v1/out_771.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260001/908C338A-AA5C-1142-8413-F4F392C85019.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018B-UL2018_MiniAODv2-v1/out_5.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260001/BFE885FB-467B-B34D-8CCC-13CAE2EBBE97.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018B-UL2018_MiniAODv2-v1/out_119.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260001/E2ED1874-3D50-EF44-914F-4CE6DE210718.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018B-UL2018_MiniAODv2-v1/out_118.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260000/8261CD8F-4724-2745-B5E0-20E52C30454E.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018B-UL2018_MiniAODv2-v1/out_770.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260000/D2170A2A-1C22-6443-88C2-502B6028B7E0.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018B-UL2018_MiniAODv2-v1/out_773.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260000/F50EC3E7-5F91-AC45-9D03-F80CFE484841.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018B-UL2018_MiniAODv2-v1/out_772.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260002/0016A9AE-CB2F-D84F-BE8F-69E4366FE967.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018B-UL2018_MiniAODv2-v1/out_1001.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260000/3D5AF3BD-5EA0-2440-97AF-33AA47D17C87.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018B-UL2018_MiniAODv2-v1/out_429.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260000/DEE30486-6A00-9449-9C3F-3F0F88C49938.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018B-UL2018_MiniAODv2-v1/out_428.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260000/37EC905D-CAA7-9A4B-8327-A9492BFC9ECD.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018B-UL2018_MiniAODv2-v1/out_534.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260001/A77FADC3-1569-7845-8853-E003419279F7.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018B-UL2018_MiniAODv2-v1/out_919.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260001/88B93A3C-9B7D-0544-B748-BCF3CC8E5771.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018B-UL2018_MiniAODv2-v1/out_918.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260000/B9546C04-5503-3C43-A262-784E57C9BF61.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018B-UL2018_MiniAODv2-v1/out_915.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260000/D298FFF9-C10F-BD4F-8596-E7BAB614F93A.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018B-UL2018_MiniAODv2-v1/out_914.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260001/64125F99-68DC-2C48-A2B2-617B806D37FD.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018B-UL2018_MiniAODv2-v1/out_917.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260000/910DBE24-1C96-6A43-A4F6-EC2E749DD4C9.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018B-UL2018_MiniAODv2-v1/out_916.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260000/4BB973F9-BC0F-C949-8A0F-A1BA7666F0F8.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018B-UL2018_MiniAODv2-v1/out_911.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260000/21C69B14-5476-4746-8364-522701303A3F.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018B-UL2018_MiniAODv2-v1/out_910.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260000/EB0D40D9-D4EE-7646-9333-2DEC0FD903A8.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018B-UL2018_MiniAODv2-v1/out_913.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260000/73D41B1A-20BA-0D49-9300-64B234761032.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018B-UL2018_MiniAODv2-v1/out_912.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260000/92860791-60A2-2E4A-9D1A-F86A37E3F3C9.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018B-UL2018_MiniAODv2-v1/out_308.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260000/D06BF6E5-3C7B-A34F-87E8-6B8579ACC108.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018B-UL2018_MiniAODv2-v1/out_309.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260000/D0F518EA-6F4F-1F4A-8EC5-E4830A20EAF3.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018B-UL2018_MiniAODv2-v1/out_855.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260000/65464D77-48AE-5C4D-8463-AB5C0B636BBA.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018B-UL2018_MiniAODv2-v1/out_300.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260000/9993FD43-249D-574B-A5AD-8CA3DCC6753D.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018B-UL2018_MiniAODv2-v1/out_301.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260000/036D8E17-6F11-6049-8B8D-22B0D188CEDD.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018B-UL2018_MiniAODv2-v1/out_302.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260000/90D4D212-A2D5-A147-BC87-6BA8F65C861E.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018B-UL2018_MiniAODv2-v1/out_303.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260000/117647E6-F236-4243-92D0-53622ACD1E7C.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018B-UL2018_MiniAODv2-v1/out_304.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260000/D90AD066-5652-DC44-BCAA-0E0A8F143C67.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018B-UL2018_MiniAODv2-v1/out_305.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260000/769E6BA2-2560-6548-B58F-62D8650CAFAF.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018B-UL2018_MiniAODv2-v1/out_306.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260000/28D12ED6-C400-EF49-8FB8-92E6CD94D7C9.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018B-UL2018_MiniAODv2-v1/out_307.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260000/C6A2CC0B-41A5-FC4D-A671-65DD152E99A2.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018B-UL2018_MiniAODv2-v1/out_895.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260000/1E78B0B4-0F37-0F4A-996C-0276DDE34CAC.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018B-UL2018_MiniAODv2-v1/out_825.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260000/6BA0811A-E193-7047-B92D-6551C5D0E971.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018B-UL2018_MiniAODv2-v1/out_665.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260000/3313A4FE-6B49-714F-AA9E-9EE11B5C82B7.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018B-UL2018_MiniAODv2-v1/out_824.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260000/F761FCD9-E857-1947-80A2-88F59456A7EC.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018B-UL2018_MiniAODv2-v1/out_649.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260000/8232ECD5-74F4-3C4D-B4FE-31021BC36880.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018B-UL2018_MiniAODv2-v1/out_847.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260000/2B93BA37-130F-D348-AEE6-817F3C988D79.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018B-UL2018_MiniAODv2-v1/out_846.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260000/4E83C1F0-45EC-E24D-80E4-2D21DD997ECF.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018B-UL2018_MiniAODv2-v1/out_845.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260000/49DB47E1-3E39-7045-9132-90C4F7A259DE.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018B-UL2018_MiniAODv2-v1/out_844.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260000/C7260631-62CD-7F43-98C1-30EFBD11616A.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018B-UL2018_MiniAODv2-v1/out_843.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260000/0F98F5F6-1528-FE4F-A616-BED1C196D609.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018B-UL2018_MiniAODv2-v1/out_842.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260000/F35A0B40-3244-CB4E-80BC-E2495B9347F0.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018B-UL2018_MiniAODv2-v1/out_841.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260000/B05F140D-A982-4E4B-8C38-2DA6CF418CF2.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018B-UL2018_MiniAODv2-v1/out_840.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260002/958843E2-8100-944F-ADC1-5D39FF1E57C3.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018B-UL2018_MiniAODv2-v1/out_1151.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260000/665EB026-0601-204E-870E-78ED1EE99DA8.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018B-UL2018_MiniAODv2-v1/out_821.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260000/D63971A8-1DF8-EA42-B8AB-BB77CF5F25F7.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018B-UL2018_MiniAODv2-v1/out_853.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260002/32F14543-D61F-CC45-B73D-E2ABAED1CAC6.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018B-UL2018_MiniAODv2-v1/out_1126.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260000/674E24DE-C0D6-8648-A810-253995913781.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018B-UL2018_MiniAODv2-v1/out_848.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260000/9B7C61DD-5786-9E46-A47E-6A831D59E52A.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018B-UL2018_MiniAODv2-v1/out_823.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260000/AFCD4CE1-06F2-A14C-B435-5ACBB4547D54.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018B-UL2018_MiniAODv2-v1/out_1209.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260000/08805769-8713-CD46-B747-F117E9B982EA.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018B-UL2018_MiniAODv2-v1/out_822.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260002/5F9BBA6A-8B99-E649-BD76-6887A76AC63B.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018B-UL2018_MiniAODv2-v1/out_1127.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260001/DCACB52E-3E6D-E140-9359-7FB0D54B5FF5.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018B-UL2018_MiniAODv2-v1/out_240.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260000/6AF76B67-6651-FC4C-922A-F970EFFE3C07.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018B-UL2018_MiniAODv2-v1/out_1208.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260001/C00A9093-C056-7649-B621-C86F30AC47F8.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018B-UL2018_MiniAODv2-v1/out_485.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260001/0B750FBE-B2A8-D344-BD46-3EBF8763115E.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018B-UL2018_MiniAODv2-v1/out_390.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260000/F8F59AB5-0822-2F47-8AF7-D6E47CFA496F.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018B-UL2018_MiniAODv2-v1/out_1038.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260001/7B98463B-1693-B548-92FC-D92E14E53345.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018B-UL2018_MiniAODv2-v1/out_732.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260000/15940B2F-331E-514E-BB35-4B68CA9DDA16.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018B-UL2018_MiniAODv2-v1/out_162.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260002/6BB33EA5-29AE-5544-A913-7FBFB3D77408.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018B-UL2018_MiniAODv2-v1/out_1166.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260001/EAD1D6D2-051C-D64C-87FC-B1DF629ACF60.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018B-UL2018_MiniAODv2-v1/out_391.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260000/7847F7B3-44D7-864C-B5C4-2C89603983BD.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018B-UL2018_MiniAODv2-v1/out_753.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260000/FD360CF9-8BC7-D240-96CC-34281B16A7BF.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018B-UL2018_MiniAODv2-v1/out_752.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260000/8A2F7567-AA32-1443-91BC-17C62B7BF4D1.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018B-UL2018_MiniAODv2-v1/out_751.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260000/BD4AE9AF-49A9-C14C-AE60-19AD256CBBE8.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018B-UL2018_MiniAODv2-v1/out_750.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260000/593219AA-DB11-8045-B030-46C3B312E56A.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018B-UL2018_MiniAODv2-v1/out_757.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260000/3A2B76B3-94FA-944A-AC80-BCC9ED863C09.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018B-UL2018_MiniAODv2-v1/out_756.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260000/3FA571EA-795C-EC4F-AB5F-57D006B1B778.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018B-UL2018_MiniAODv2-v1/out_755.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260000/3F17E1FD-EBCE-9E4D-A458-EFE6051EF523.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018B-UL2018_MiniAODv2-v1/out_754.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260000/645BDB27-3141-CB43-837F-93B32592B130.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018B-UL2018_MiniAODv2-v1/out_560.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260000/4E724565-284C-B243-8279-0C9DBCB11464.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018B-UL2018_MiniAODv2-v1/out_561.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260000/D1CA210F-D2DB-A347-84D4-67CF98541702.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018B-UL2018_MiniAODv2-v1/out_759.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260000/A94BBB9D-CE9E-6F4E-BAEE-0E9CDD0E110E.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018B-UL2018_MiniAODv2-v1/out_758.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260000/A181E19A-4F30-6244-8B47-5044D0D30D3E.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018B-UL2018_MiniAODv2-v1/out_564.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260000/1FF0FF35-EBAB-BC4E-B609-F5F07E3D9A9B.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018B-UL2018_MiniAODv2-v1/out_565.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260000/4BFDBF76-E139-5847-AE4A-45DCA3F096BB.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018B-UL2018_MiniAODv2-v1/out_566.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260000/75F1AB98-24DB-AF4E-AAE4-5C986906F626.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018B-UL2018_MiniAODv2-v1/out_701.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260000/FF054269-D276-E644-9C92-7EBED0F8F4BC.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018B-UL2018_MiniAODv2-v1/out_506.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260000/576268AA-ADD3-B043-A7D9-90B2BE06AF9B.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018B-UL2018_MiniAODv2-v1/out_229.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260000/517B158F-E1B8-A241-8C61-B1FD4F63F49B.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018B-UL2018_MiniAODv2-v1/out_507.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260000/65788115-D977-954E-B1B9-5E5F3FC7E3AC.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018B-UL2018_MiniAODv2-v1/out_227.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260000/CBCDA58B-8D2E-D040-BCCB-A0D540356BCB.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018B-UL2018_MiniAODv2-v1/out_226.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260000/89AB6B94-0C6B-C946-8224-8F52C9F25C46.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018B-UL2018_MiniAODv2-v1/out_225.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260001/94D3683A-0299-954B-9A19-7390BCF77276.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018B-UL2018_MiniAODv2-v1/out_985.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260000/83CD2FCA-C7DD-0D4A-9AC8-A0C0DAF690AC.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018B-UL2018_MiniAODv2-v1/out_223.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260000/92C29CA4-8D2E-C64F-BDC8-05448EBE6FB4.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018B-UL2018_MiniAODv2-v1/out_504.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260000/0D8AC1E3-E193-3F4A-A6D5-8B2D1682C6BD.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018B-UL2018_MiniAODv2-v1/out_221.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260000/FB1DFBB2-5561-0D4C-A4E8-5C595C9B990A.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018B-UL2018_MiniAODv2-v1/out_220.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260002/9E3D62B6-1480-D34A-9DF5-BFCF8287714A.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018B-UL2018_MiniAODv2-v1/out_1025.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260002/212AC708-B853-414D-85B3-49EDA2C6D90B.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018B-UL2018_MiniAODv2-v1/out_1024.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260002/40F5F179-89A4-5F4E-8911-FB4973FEB1E9.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018B-UL2018_MiniAODv2-v1/out_1027.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260002/1DBE915A-7DDF-F346-AE19-708DB0B8213B.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018B-UL2018_MiniAODv2-v1/out_1031.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260000/ED6D63C1-44DC-2E4C-A06F-FB705F1E0193.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018B-UL2018_MiniAODv2-v1/out_1021.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260000/07E95A01-502F-BF4F-AC57-0337A9D545DA.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018B-UL2018_MiniAODv2-v1/out_1020.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260002/2FF211A4-09C1-6C48-9EEE-18E872B9F685.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018B-UL2018_MiniAODv2-v1/out_1023.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260002/F21F3032-4FF8-A541-BA2F-3681FA7B319A.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018B-UL2018_MiniAODv2-v1/out_1189.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260002/5D6A9CA6-4F57-0D49-81BC-11FDBD612102.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018B-UL2018_MiniAODv2-v1/out_1186.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260002/39C7B33F-FC33-6F44-B49C-BCE7CFD2AEB2.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018B-UL2018_MiniAODv2-v1/out_1187.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260000/82DF76B0-3857-0A4C-932F-C85CDE27FEE4.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018B-UL2018_MiniAODv2-v1/out_1184.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260002/B4777C23-3745-094A-B0FD-D72206C1B2E2.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018B-UL2018_MiniAODv2-v1/out_1185.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260002/31472619-2437-454E-BB59-1C2F181B3992.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018B-UL2018_MiniAODv2-v1/out_1182.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260000/A5487F76-9B09-5E4C-B23B-88827297324D.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018B-UL2018_MiniAODv2-v1/out_1183.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260002/D0365C86-FA3E-BA4E-8B21-742B1E43DA18.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018B-UL2018_MiniAODv2-v1/out_1180.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260002/87268789-BBD0-B44A-8BD6-E1734C7B4E2A.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018B-UL2018_MiniAODv2-v1/out_1181.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260001/02BB2EA5-3934-B640-B600-6A231396EBC1.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018B-UL2018_MiniAODv2-v1/out_726.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260001/9F5A5C4A-B884-9D45-868A-64CC82FB990D.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018B-UL2018_MiniAODv2-v1/out_727.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260001/EBA78DBD-BB97-9D48-B9A2-03E0604324C6.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018B-UL2018_MiniAODv2-v1/out_724.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260000/4F55BBA1-3F84-4E4D-A13F-70881E79BF93.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018B-UL2018_MiniAODv2-v1/out_89.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260000/1A49291D-449C-3140-9EA5-9C6284FC9990.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018B-UL2018_MiniAODv2-v1/out_568.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260002/8EF05A51-622E-774A-BAFF-827FD2AF8F0C.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018B-UL2018_MiniAODv2-v1/out_1034.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260001/EC1B0595-6A04-0D4A-A31B-20763F51B138.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018B-UL2018_MiniAODv2-v1/out_720.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260001/C773CE79-47B0-284E-ACED-33CA5D483A23.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018B-UL2018_MiniAODv2-v1/out_721.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260002/477E2B99-5077-6444-B4C5-D3B708AD1598.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018B-UL2018_MiniAODv2-v1/out_1160.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260000/B46D2D64-DB98-454B-AA42-D1B09113169D.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018B-UL2018_MiniAODv2-v1/out_1035.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260002/215C7A00-F9F7-9B4F-B886-1ACAA3307464.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018B-UL2018_MiniAODv2-v1/out_1167.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260001/416EF6F2-4529-2549-9B13-849026D371E2.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018B-UL2018_MiniAODv2-v1/out_728.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260001/3CD48547-53AD-A44B-9B77-1A75C8F7E5B6.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018B-UL2018_MiniAODv2-v1/out_729.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260001/607B93E9-E373-F943-8F0C-0E6822ECF563.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018B-UL2018_MiniAODv2-v1/out_605.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260001/5FE6BC5E-1C20-9742-83CE-3AE138E9768D.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018B-UL2018_MiniAODv2-v1/out_604.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260001/43559296-7A56-244D-9F27-B59F0414F2F6.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018B-UL2018_MiniAODv2-v1/out_607.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260001/1D56955B-F053-524C-972F-D8E2474B8897.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018B-UL2018_MiniAODv2-v1/out_606.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260001/4A115AA2-41F4-7C4B-B03B-8B1D24D57049.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018B-UL2018_MiniAODv2-v1/out_601.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260001/8E110291-AD3F-604A-9409-3EBBCFC44651.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018B-UL2018_MiniAODv2-v1/out_600.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260002/2DD6ED97-5284-5D47-8272-604D7E4453C6.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018B-UL2018_MiniAODv2-v1/out_1162.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260001/EDFD2878-A952-EB42-827D-4F2D37CAEC1C.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018B-UL2018_MiniAODv2-v1/out_602.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260002/F9F6F0E2-4B63-5F42-B79C-2BA3E7BFCB61.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018B-UL2018_MiniAODv2-v1/out_1205.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260000/8768107B-FB07-CB40-9562-59D1D5946E9C.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018B-UL2018_MiniAODv2-v1/out_158.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260000/DFCB503E-7B6F-7143-A7AF-8CDE6997585F.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018B-UL2018_MiniAODv2-v1/out_1207.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260001/21DCCE9B-D5C6-B346-ABDD-F75AEE27DC15.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018B-UL2018_MiniAODv2-v1/out_112.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260002/01F061C9-22D9-EE40-8FCD-65A7CDA15092.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018B-UL2018_MiniAODv2-v1/out_1168.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260002/535B2C17-6D11-A342-82EE-F06F37B14419.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018B-UL2018_MiniAODv2-v1/out_1200.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260002/1A0BD35F-C3F8-684E-BA56-B7610965DFCB.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018B-UL2018_MiniAODv2-v1/out_1203.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260002/37C37C33-B7B4-4A41-8D19-6B19207862EE.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018B-UL2018_MiniAODv2-v1/out_1202.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260002/C6495DE4-C9AF-D44C-9378-9F694401CD4F.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018B-UL2018_MiniAODv2-v1/out_1204.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260002/68DDAE73-859E-5B42-9DEB-B220CA1DA9F7.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018B-UL2018_MiniAODv2-v1/out_1211.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260001/E1161473-0A3E-A04D-A77A-9908523F84A9.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018B-UL2018_MiniAODv2-v1/out_976.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260002/C9449F49-2BFD-2948-8C3B-454840081040.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018B-UL2018_MiniAODv2-v1/out_1161.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260000/4E83BA17-A8CC-3A49-84C5-1B95FE9CECCB.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018B-UL2018_MiniAODv2-v1/out_469.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260001/642B7BFD-603E-294F-82D2-5BF9AC3D6B40.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018B-UL2018_MiniAODv2-v1/out_935.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260001/390A8B89-81DF-2545-9EEE-0B96B036ADC5.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018B-UL2018_MiniAODv2-v1/out_399.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260002/75822584-118F-5548-A93F-408769621FE0.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018B-UL2018_MiniAODv2-v1/out_1217.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260001/18F3FF7F-ECB4-924B-B4DA-9CA0220D820D.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018B-UL2018_MiniAODv2-v1/out_959.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260000/4BFD7347-4732-844C-8301-4CD5AB1B6ECB.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018B-UL2018_MiniAODv2-v1/out_958.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260001/C639FA6A-0943-0D47-A6FE-1B03FA36C022.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018B-UL2018_MiniAODv2-v1/out_398.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260000/2E90594A-1197-FC4D-AE11-351382339C16.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018B-UL2018_MiniAODv2-v1/out_1214.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260001/31A952CE-7C8A-394A-BEAC-C2D79983008B.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018B-UL2018_MiniAODv2-v1/out_48.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260001/F6EF9CE4-417C-674A-B4F2-8B36BA4A980F.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018B-UL2018_MiniAODv2-v1/out_49.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260001/91F36FFD-EC14-D040-AEDC-D8650653CD0C.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018B-UL2018_MiniAODv2-v1/out_951.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260001/1F70D806-E289-FE46-82BF-A2237FD5FEBF.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018B-UL2018_MiniAODv2-v1/out_950.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260000/6DF0166A-8FF3-F949-AA87-5DFF8CECBD92.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018B-UL2018_MiniAODv2-v1/out_953.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260000/A13F2DB3-24DC-A940-8038-8E54A66C941B.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018B-UL2018_MiniAODv2-v1/out_952.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260000/FD942232-675F-444D-8643-C72B4A8B44C8.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018B-UL2018_MiniAODv2-v1/out_42.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260000/1D4ECF2B-58F1-E747-AEB9-842332BBF855.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018B-UL2018_MiniAODv2-v1/out_954.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260000/3CB94050-CFDD-9E45-9455-766DA65C95BB.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018B-UL2018_MiniAODv2-v1/out_957.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260000/65B7E98B-A8D1-C048-A2D2-CE146636E774.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018B-UL2018_MiniAODv2-v1/out_956.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260002/36B36B65-DAC7-5044-A69C-ADC05342FEE9.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018B-UL2018_MiniAODv2-v1/out_1111.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260002/50DC6A00-5327-DE4A-841C-824509A95361.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018B-UL2018_MiniAODv2-v1/out_1110.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260002/76F5E256-41CB-7649-84EA-1CCEE0F757DB.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018B-UL2018_MiniAODv2-v1/out_1113.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260002/F4BF5577-8235-C247-8CE7-3F981762DD6C.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018B-UL2018_MiniAODv2-v1/out_1112.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260002/6004F141-852C-CC4B-BF18-89D3E6621CE7.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018B-UL2018_MiniAODv2-v1/out_1115.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260002/EF55D7FD-8B88-AB43-BF56-EF36F1C6AB5D.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018B-UL2018_MiniAODv2-v1/out_1114.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260002/0CC42B11-87CF-724D-948F-66DFA098A9A7.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018B-UL2018_MiniAODv2-v1/out_1117.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260002/0FA3243E-4439-E54E-9AE6-E8BF0349CDF3.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018B-UL2018_MiniAODv2-v1/out_1116.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260001/C35811B2-DEB7-B540-8C01-2569E7700925.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018B-UL2018_MiniAODv2-v1/out_620.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260001/B2FBF0ED-6D8A-D24B-A56D-E96A8BF14BF9.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018B-UL2018_MiniAODv2-v1/out_608.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260000/C4BA5586-FBE6-014F-BD02-5723DDC6BDE5.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018B-UL2018_MiniAODv2-v1/out_467.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260002/C95579EF-E62E-DE44-9339-1754B2A902D8.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018B-UL2018_MiniAODv2-v1/out_1188.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260002/B71862E2-7EDB-9D4C-B041-D0DC0483E94B.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018B-UL2018_MiniAODv2-v1/out_1219.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260000/70A53659-80ED-1040-A8FE-9B67EA3D78A1.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018B-UL2018_MiniAODv2-v1/out_1206.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260001/9BF74A5C-F30F-A34C-AF47-D6CA65EDAC6F.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018B-UL2018_MiniAODv2-v1/out_460.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260002/3E8DC06F-C0E1-EE42-BE2B-7EEA8841EB15.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018B-UL2018_MiniAODv2-v1/out_1163.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260000/E7481913-C78A-FB44-B4F4-4BC9070D0F33.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018B-UL2018_MiniAODv2-v1/out_489.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260000/D9AFF144-5D9D-AA4A-94B7-B23E2503B21B.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018B-UL2018_MiniAODv2-v1/out_488.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260000/D33609CC-2505-F441-BF42-A4375ECB8C3B.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018B-UL2018_MiniAODv2-v1/out_487.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260000/F97F2A99-9320-D846-8E9A-96D2C58DF0FD.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018B-UL2018_MiniAODv2-v1/out_486.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260001/96DB7BA2-B45A-3042-A6D6-034128D8369E.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018B-UL2018_MiniAODv2-v1/out_869.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260001/C371A52A-171C-2644-9D28-566AC71740F9.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018B-UL2018_MiniAODv2-v1/out_484.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260001/39780868-2204-8744-BF14-5CF49D5EBC03.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018B-UL2018_MiniAODv2-v1/out_483.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260001/57D38FA7-42CF-124A-AE8A-3CECE3A7EF80.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018B-UL2018_MiniAODv2-v1/out_482.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260001/19A9F9AA-D201-994B-BC25-63AC344E7F03.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018B-UL2018_MiniAODv2-v1/out_481.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260001/9C547CC5-CFD5-C249-8CBA-9D61EDE55684.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018B-UL2018_MiniAODv2-v1/out_480.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260000/BA235A2F-7EDE-F748-B1B3-7583DFCA6607.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018B-UL2018_MiniAODv2-v1/out_509.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260000/23BCB24F-1AC3-874A-AD70-9838B2AABC9D.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018B-UL2018_MiniAODv2-v1/out_955.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260002/D6B2EF7D-5C52-7947-934B-4DB787CFA463.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018B-UL2018_MiniAODv2-v1/out_1201.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260001/53F777EB-60A0-734C-86BF-475A67EFCC81.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018B-UL2018_MiniAODv2-v1/out_472.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260001/1E9CFC79-3E18-6C49-B79F-BCABDB3D20C3.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018B-UL2018_MiniAODv2-v1/out_473.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260000/C8DB74A4-909B-DC44-B5F2-7F0C2104199B.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018B-UL2018_MiniAODv2-v1/out_470.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260001/2D69B59C-AC4F-DD4A-A6FE-2BEA0E77109A.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018B-UL2018_MiniAODv2-v1/out_471.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260001/F0CE94C5-E2BE-9D4B-BEB2-B72A24E2B537.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018B-UL2018_MiniAODv2-v1/out_476.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260001/405149EE-3A87-8B45-876D-8DBD2BA86699.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018B-UL2018_MiniAODv2-v1/out_477.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260001/7E11AAEA-97D5-BC4D-B8F4-D5FB4E50969D.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018B-UL2018_MiniAODv2-v1/out_474.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260001/679F5DA7-DA3A-564E-AEF8-D0642B8A1122.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018B-UL2018_MiniAODv2-v1/out_475.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260001/182FCEF9-AA68-C947-B47B-E1479354DC3F.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018B-UL2018_MiniAODv2-v1/out_46.root -a '-o ./ -d --year UL2018B'"
queue
arguments="-f /store/data/Run2018B/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260001/D454FB54-203A-8843-8DD5-AB2DC1E87090.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018B-UL2018_MiniAODv2-v1/out_479.root -a '-o ./ -d --year UL2018B'"
queue
