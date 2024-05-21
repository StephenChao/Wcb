universe=Vanilla
+ProjectName = "cms.org"
requirements = (OpSysAndVer =?= "CentOS7")
RequestMemory = 2048
RequestCpus = 1
executable=/home/legao/Wcb/condor/production/NanoNtupleChain_25_Apr_2024/SingleMuon_Run2018C-UL2018_MiniAODv2-v2/exe_CUSTNANO_UL18_MINIAODv2_DATA.sh
transfer_executable=True
transfer_input_files=/home/legao/Wcb/condor/x509up_u14433,/home/legao/Wcb/condor/Scripts/fetchFiles.py
transfer_output_files=""
log=/home/legao/Wcb/condor/production/NanoNtupleChain_25_Apr_2024/SingleMuon_Run2018C-UL2018_MiniAODv2-v2/log//$(Cluster).log
output=/home/legao/Wcb/condor/production/NanoNtupleChain_25_Apr_2024/SingleMuon_Run2018C-UL2018_MiniAODv2-v2/log//$(Cluster).out
error=/home/legao/Wcb/condor/production/NanoNtupleChain_25_Apr_2024/SingleMuon_Run2018C-UL2018_MiniAODv2-v2/log//$(Cluster).err
notification=Never
should_transfer_files = YES
when_to_transfer_output = ON_EXIT
x509userproxy=x509up_u14433
+MaxRuntime=150000
on_exit_remove        = (ExitBySignal == False) && (ExitCode == 0)
on_exit_hold          = (ExitBySignal == True) || (ExitCode != 0)
on_exit_hold_reason   = strcat("Job held by ON_EXIT_HOLD due to ", ifThenElse((ExitBySignal == True), strcat("exit signal ", ExitSignal), strcat("exit code ", ExitCode)), ".")
periodic_release      = (NumJobStarts < 3) && ((CurrentTime - EnteredCurrentStatus) > 60*60)
arguments="-f /store/data/Run2018C/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/EF40B1A0-1C86-0643-B264-8E189316CE61.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018C-UL2018_MiniAODv2-v2/out_723.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/66000D46-D02D-D346-8B1C-86B2B32F3E3D.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018C-UL2018_MiniAODv2-v2/out_198.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/74BC1B5B-2182-C64B-8600-EFE4AE40C06D.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018C-UL2018_MiniAODv2-v2/out_669.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/2DAF620B-7A2B-FB4C-BE6E-B8AAA2B9AE2A.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018C-UL2018_MiniAODv2-v2/out_668.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/0D689E84-F545-6845-A003-A7915B1BF514.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018C-UL2018_MiniAODv2-v2/out_344.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/ED6CFEEE-6960-A94C-8EAE-1A2D5430176C.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018C-UL2018_MiniAODv2-v2/out_819.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/CE577D18-19B1-C747-85B5-00C68ADF9211.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018C-UL2018_MiniAODv2-v2/out_346.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/23038C27-BEA5-C746-9EB5-681743B8CCE2.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018C-UL2018_MiniAODv2-v2/out_347.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/CF878F7D-00C9-E745-8FDD-0A0C851C73BE.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018C-UL2018_MiniAODv2-v2/out_340.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/062CB925-7FC1-0D42-842D-6CE64B584FC5.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018C-UL2018_MiniAODv2-v2/out_341.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/181A8824-C65D-EA46-B875-13F3BB679788.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018C-UL2018_MiniAODv2-v2/out_342.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/BA3211BD-ABF3-144C-AF7F-95FD6DE2F07A.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018C-UL2018_MiniAODv2-v2/out_343.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/58897C7F-B743-E843-924B-8DC42FE71448.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018C-UL2018_MiniAODv2-v2/out_810.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/6CE2D014-1C19-4946-B3CF-3A120EB3B7E3.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018C-UL2018_MiniAODv2-v2/out_811.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/B75A585E-9880-BD44-B890-F73821B43BDF.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018C-UL2018_MiniAODv2-v2/out_812.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/467C2FDE-6921-414E-A161-7B7A3B16F1E0.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018C-UL2018_MiniAODv2-v2/out_813.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/E491D429-9C78-1141-9BA0-B0112A6C5487.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018C-UL2018_MiniAODv2-v2/out_348.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70002/7E0B5E69-D78D-994D-AB26-F6778D0355C1.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018C-UL2018_MiniAODv2-v2/out_349.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/32C96987-8601-1F45-8C43-0B878A10418F.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018C-UL2018_MiniAODv2-v2/out_816.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/0D3C7AE1-BAF3-F347-A2A9-6E801CF657C6.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018C-UL2018_MiniAODv2-v2/out_817.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/83222223-3113-0849-B36C-03A337B8667A.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018C-UL2018_MiniAODv2-v2/out_664.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/7E8C3558-56F7-DF4A-A36D-BEFC84F237C3.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018C-UL2018_MiniAODv2-v2/out_739.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/DC82527E-B4B2-2C46-AF85-0E744390480D.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018C-UL2018_MiniAODv2-v2/out_719.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/C312D830-0879-4342-A18B-24DCDB07884D.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018C-UL2018_MiniAODv2-v2/out_718.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/C38912E0-69DB-114B-8D8D-4F4D10DEBADE.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018C-UL2018_MiniAODv2-v2/out_717.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/2B679F3D-AEB8-114E-B334-1F44B5AF7771.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018C-UL2018_MiniAODv2-v2/out_716.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/4FF15BF6-4C88-D243-8325-2E574AD59A1F.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018C-UL2018_MiniAODv2-v2/out_715.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/FC838FEA-0322-314C-BFE3-B5DD47DA23AC.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018C-UL2018_MiniAODv2-v2/out_714.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/CA1BCE13-65C6-5240-AE8F-49B5FE477310.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018C-UL2018_MiniAODv2-v2/out_713.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/DA38F905-81FA-7040-93F7-B83558FE538A.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018C-UL2018_MiniAODv2-v2/out_712.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/68A9C2F9-0679-FD4C-90EA-B20FF6444FA3.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018C-UL2018_MiniAODv2-v2/out_711.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/4CF9DAC0-B695-654C-8485-521B4061488D.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018C-UL2018_MiniAODv2-v2/out_710.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/C091AB92-9C6C-B146-A081-82EB7A71C74E.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018C-UL2018_MiniAODv2-v2/out_618.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70002/A12575D4-6995-7648-9EA7-ABC8EF9A2B49.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018C-UL2018_MiniAODv2-v2/out_421.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70001/3F2AF554-9769-2546-97C1-C4832256D727.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018C-UL2018_MiniAODv2-v2/out_129.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/CDEEC36F-20F5-284E-8B8D-4A0FF2124A18.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018C-UL2018_MiniAODv2-v2/out_420.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70001/1E94AA45-4316-CA4C-A5FC-A8198673A72A.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018C-UL2018_MiniAODv2-v2/out_423.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/2EFA6E00-A360-D649-83FE-74211BE841ED.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018C-UL2018_MiniAODv2-v2/out_1069.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/50C85CE3-CEDA-5248-B1AA-4664A781AA41.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018C-UL2018_MiniAODv2-v2/out_1068.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/82527765-5CF6-8548-A532-97421998020F.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018C-UL2018_MiniAODv2-v2/out_422.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/F5A7F1A1-9160-0E46-8FBE-AF63394E897F.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018C-UL2018_MiniAODv2-v2/out_1061.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70001/AA0D0F31-04FD-904E-A517-3D84615C88DA.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018C-UL2018_MiniAODv2-v2/out_1060.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/925BB478-9760-E245-860B-AB63502A0D95.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018C-UL2018_MiniAODv2-v2/out_1063.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/1280EC91-C30C-5C41-A2CF-8F2D38BFC03A.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018C-UL2018_MiniAODv2-v2/out_1062.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/3AC8585E-B1E1-7F4A-8C07-BD75E605BB3E.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018C-UL2018_MiniAODv2-v2/out_1065.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/92B4427D-CBFE-FB45-B8DF-69DDD0796E1D.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018C-UL2018_MiniAODv2-v2/out_1064.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/53757A6C-718E-3D4D-9C4C-A919A155A2F6.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018C-UL2018_MiniAODv2-v2/out_1067.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/E6918811-7BE1-5F4E-BD20-7A2F71E27929.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018C-UL2018_MiniAODv2-v2/out_1066.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/29C9A3B4-0CC2-BE42-9E91-7FBF2EA91052.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018C-UL2018_MiniAODv2-v2/out_671.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/2196309A-005E-9148-ADF9-358A8F27014E.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018C-UL2018_MiniAODv2-v2/out_427.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/C7B0B531-40B4-1640-B89D-0473C5D86142.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018C-UL2018_MiniAODv2-v2/out_298.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/DF2A7DA0-524D-D340-87D5-4CFBE34194B0.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018C-UL2018_MiniAODv2-v2/out_299.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/AB4E1A5E-AD78-3A4F-BDDC-6C8F78313FFB.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018C-UL2018_MiniAODv2-v2/out_296.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/88BD330C-0E83-AB46-ADC8-9F69DF1F01F8.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018C-UL2018_MiniAODv2-v2/out_297.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70002/DDAF2124-DCEA-604F-9914-C343FB6F5B79.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018C-UL2018_MiniAODv2-v2/out_294.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70002/7C9834BB-1C11-4E40-8441-377F41891C35.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018C-UL2018_MiniAODv2-v2/out_295.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70002/2D3087AC-D796-6D48-BECC-E12E05590AAB.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018C-UL2018_MiniAODv2-v2/out_292.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70002/DF415D80-8CA4-6544-A1F1-6660F90C0195.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018C-UL2018_MiniAODv2-v2/out_293.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70002/86771DAF-137F-5949-A5C1-8A481DC695FA.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018C-UL2018_MiniAODv2-v2/out_290.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70002/6FBAFD72-C7F8-D048-A539-1CDD3CCFD206.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018C-UL2018_MiniAODv2-v2/out_291.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/C81F8D35-E6CB-2449-B4AB-8DFB7AD63556.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018C-UL2018_MiniAODv2-v2/out_591.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/2D586564-92FE-424C-BB14-7F7E51A5FEF2.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018C-UL2018_MiniAODv2-v2/out_590.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/5B744589-E7B4-024B-89CA-D9A6D3E11DE2.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018C-UL2018_MiniAODv2-v2/out_593.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/59058E94-AB05-324C-9093-5C0C0FA95E22.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018C-UL2018_MiniAODv2-v2/out_592.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/BC774D4C-CB20-F147-8F50-A0B6F1FF5809.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018C-UL2018_MiniAODv2-v2/out_595.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/05852D2C-8B42-AA46-9C8C-0EEBEE4C8D58.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018C-UL2018_MiniAODv2-v2/out_1082.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70002/8CA589E3-44B7-4847-ABD3-570005C6F628.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018C-UL2018_MiniAODv2-v2/out_597.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/3B95AC2A-CFAD-EC44-8610-47528AB8324B.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018C-UL2018_MiniAODv2-v2/out_596.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/9537B568-823A-CC4C-9CFC-A2D36AB576A7.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018C-UL2018_MiniAODv2-v2/out_195.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/F20E2222-82E2-2248-8785-2271B2FD4B33.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018C-UL2018_MiniAODv2-v2/out_194.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/59C96777-7654-BE46-A87C-8634024D8682.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018C-UL2018_MiniAODv2-v2/out_197.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/02CE901D-C74D-EB47-98FA-BAAB6FEAA5BA.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018C-UL2018_MiniAODv2-v2/out_196.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/0CFE8AB0-55CB-9847-847D-A3533FA146D2.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018C-UL2018_MiniAODv2-v2/out_191.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/43FE4141-ACFF-E643-BD61-9E0B766E1E4E.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018C-UL2018_MiniAODv2-v2/out_190.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/58D342CE-5270-E24D-B500-D57806BA264B.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018C-UL2018_MiniAODv2-v2/out_193.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/36CA9F5A-3FA6-A64C-82B6-859CAEDB7B83.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018C-UL2018_MiniAODv2-v2/out_192.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/36C75A7B-F181-524F-8E5F-D8C6FC91E16D.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018C-UL2018_MiniAODv2-v2/out_270.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70002/D9438F7B-0A69-6044-80DB-D412E9FBE1BE.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018C-UL2018_MiniAODv2-v2/out_271.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70002/5F2544AE-BCD5-2D4F-A9A0-325535435C11.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018C-UL2018_MiniAODv2-v2/out_272.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70002/F8DE99EE-F33D-404A-BCA3-2283BB68F522.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018C-UL2018_MiniAODv2-v2/out_273.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70002/0A85F48E-0945-1944-B5F8-1FFD19890807.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018C-UL2018_MiniAODv2-v2/out_274.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70002/5DD4DF4B-A10E-2F43-B018-F82AA6FABB7E.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018C-UL2018_MiniAODv2-v2/out_275.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70002/C9689568-373A-A941-85DD-D628159CF974.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018C-UL2018_MiniAODv2-v2/out_276.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70002/343D2E01-452C-7E45-8EC1-390A84B5744E.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018C-UL2018_MiniAODv2-v2/out_277.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70002/DC24EAB8-CEE7-134C-9F29-7F31E49957E5.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018C-UL2018_MiniAODv2-v2/out_278.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70002/3357D837-5BFC-7E48-BBED-3FFA4670DF5E.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018C-UL2018_MiniAODv2-v2/out_279.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/4DEDB918-0F64-AB4D-8133-3ADAE9D00339.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018C-UL2018_MiniAODv2-v2/out_569.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/D116DBB1-4CC4-AA40-9C81-B3D9BBCAF8CD.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018C-UL2018_MiniAODv2-v2/out_497.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/3F45F714-F280-CF44-B558-1E14493B1CDB.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018C-UL2018_MiniAODv2-v2/out_524.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/8F93332A-F7BD-0840-A11D-83CC7CA39CC6.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018C-UL2018_MiniAODv2-v2/out_525.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/EF1EE85D-5664-724F-B1A7-F5A263FB0C4D.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018C-UL2018_MiniAODv2-v2/out_526.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/F7B31359-8925-DC4E-BEB7-6B053B6927A4.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018C-UL2018_MiniAODv2-v2/out_527.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/5EC50C47-21DC-F04C-A3CA-D4F208103F94.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018C-UL2018_MiniAODv2-v2/out_520.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/4A30BE54-ABD2-B54E-94D9-69EACFD31FD9.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018C-UL2018_MiniAODv2-v2/out_521.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/05B35086-865A-4647-BF86-88771F4434BA.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018C-UL2018_MiniAODv2-v2/out_522.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/3E798CB9-6335-0149-9557-AFE0813B9F9E.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018C-UL2018_MiniAODv2-v2/out_523.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/9AD2A205-7341-4C48-8EA6-A976FD4ED594.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018C-UL2018_MiniAODv2-v2/out_1014.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/C1C0C15A-D938-8148-B91D-A1E9C9ED11B1.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018C-UL2018_MiniAODv2-v2/out_1015.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/ACD91182-7563-3442-8AE8-C3366FDC5054.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018C-UL2018_MiniAODv2-v2/out_599.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/F4922705-AABE-B749-891E-FC4CC982EFCE.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018C-UL2018_MiniAODv2-v2/out_1017.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/9E9965F3-CC8B-EC4D-9342-FA60C3A6E673.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018C-UL2018_MiniAODv2-v2/out_528.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/9C22AFBC-8B1C-ED4D-BB3A-6D80AB90B050.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018C-UL2018_MiniAODv2-v2/out_529.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/F15ECAC3-6420-C94F-844A-3386673D854F.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018C-UL2018_MiniAODv2-v2/out_1012.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/2E102E13-67C5-A44B-9B6D-61FD8858C3A6.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018C-UL2018_MiniAODv2-v2/out_1013.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/5757454D-9D9E-0640-8355-CDC73CDA1F88.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018C-UL2018_MiniAODv2-v2/out_1234.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/155E33B3-2278-734A-BC9F-1A578328AFDB.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018C-UL2018_MiniAODv2-v2/out_1235.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/7EA1EFD5-9103-864E-A697-3834D09C1B10.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018C-UL2018_MiniAODv2-v2/out_1236.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/665774AA-1A57-1B46-8E8D-FFF38E6816E8.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018C-UL2018_MiniAODv2-v2/out_448.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70001/B2DF551F-1EB3-594A-A80A-6B82F83488BB.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018C-UL2018_MiniAODv2-v2/out_1230.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/CCAC450A-F47D-7748-B4D5-886E0E4BC3F2.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018C-UL2018_MiniAODv2-v2/out_1231.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/0162D441-EF71-CB48-A05C-7FE0DB8FA3A8.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018C-UL2018_MiniAODv2-v2/out_1232.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/9C1EAE3C-C936-674A-B8A7-DBBF7EF8421A.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018C-UL2018_MiniAODv2-v2/out_1233.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/115262D9-10A2-5844-81DF-A484431B0A6E.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018C-UL2018_MiniAODv2-v2/out_443.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/F6063D10-4BF9-3D44-84B5-DEBBDF7EFCC0.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018C-UL2018_MiniAODv2-v2/out_442.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/FB17163F-A65C-D74A-8683-6E723861BFD5.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018C-UL2018_MiniAODv2-v2/out_441.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/B0424E3D-BE82-924C-A51D-F51455581D62.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018C-UL2018_MiniAODv2-v2/out_440.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/F21918CE-9395-904A-BBBA-4AC093E20510.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018C-UL2018_MiniAODv2-v2/out_447.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/882A5767-3216-704B-B4A2-E01966329A26.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018C-UL2018_MiniAODv2-v2/out_446.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/06C430EA-3D1D-8246-A000-21F990FF07E8.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018C-UL2018_MiniAODv2-v2/out_445.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/28006E89-6498-DA48-B143-9CF65201C14A.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018C-UL2018_MiniAODv2-v2/out_444.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/50B4B1D9-D91B-8742-A57E-094BE27CB86B.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018C-UL2018_MiniAODv2-v2/out_47.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/DFAC8E07-1E27-9748-9405-46CE6579E091.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018C-UL2018_MiniAODv2-v2/out_108.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/D65D705F-D491-1B42-86CA-178380B9EC92.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018C-UL2018_MiniAODv2-v2/out_109.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/6278D147-BD06-0F49-BC3D-B25CB5994D40.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018C-UL2018_MiniAODv2-v2/out_588.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/ED678505-15F3-6045-BCD6-62CD35C670D5.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018C-UL2018_MiniAODv2-v2/out_102.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/8F7FCDF7-4F60-DA40-814C-CC78723BA3D7.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018C-UL2018_MiniAODv2-v2/out_103.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/CC095947-C3F3-4941-9CDD-FBF0E95AFE3D.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018C-UL2018_MiniAODv2-v2/out_100.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/C603F75D-C3CF-034A-937F-884752BB578D.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018C-UL2018_MiniAODv2-v2/out_101.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/575007BC-C28B-5444-B55B-C29D5FC2680B.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018C-UL2018_MiniAODv2-v2/out_106.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/04DF28FC-A01B-834C-8B6C-333461E07D9D.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018C-UL2018_MiniAODv2-v2/out_107.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/4EFD074A-F470-964A-8853-61E7ED979042.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018C-UL2018_MiniAODv2-v2/out_104.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/1FC8AD5D-088B-5D4D-9A46-065587A52DCE.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018C-UL2018_MiniAODv2-v2/out_105.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/A69F2B78-A20D-1841-A9C1-B73BF9C76ADA.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018C-UL2018_MiniAODv2-v2/out_902.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/F26E3555-19F8-844E-9154-BB66121C62A6.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018C-UL2018_MiniAODv2-v2/out_903.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/0D9E6AA4-FE01-6F45-8E62-FB407433211C.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018C-UL2018_MiniAODv2-v2/out_39.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/8EF0BB27-B37A-C448-96CC-7522DDA0F53E.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018C-UL2018_MiniAODv2-v2/out_38.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/F8CF613D-0010-DC48-BD68-5C168698B09D.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018C-UL2018_MiniAODv2-v2/out_906.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/563F4C69-9963-4B42-888F-1BDA2669C256.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018C-UL2018_MiniAODv2-v2/out_907.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/D7429EE2-29F1-FD45-A899-3155F8B14B23.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018C-UL2018_MiniAODv2-v2/out_904.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/01310900-85D8-7149-AD77-CF7C7A40EED1.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018C-UL2018_MiniAODv2-v2/out_905.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70002/A8DF698C-5BBC-1542-9BAA-8B7521FACE86.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018C-UL2018_MiniAODv2-v2/out_33.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70002/400A6252-A77B-C943-83A3-DAF69F9CC4EF.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018C-UL2018_MiniAODv2-v2/out_32.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/D395E8A2-F3EF-C447-A2FD-DAF30CA573A2.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018C-UL2018_MiniAODv2-v2/out_31.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/5DCFBB05-832A-9942-9783-0190CF22B0DD.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018C-UL2018_MiniAODv2-v2/out_30.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70002/56A5526C-BE21-604B-B3D5-D0AD19C6D169.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018C-UL2018_MiniAODv2-v2/out_37.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70002/006686D9-26EE-1044-8B11-22EFB164E2DE.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018C-UL2018_MiniAODv2-v2/out_36.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70002/7F3592A1-801F-6940-9627-3BE9895E3852.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018C-UL2018_MiniAODv2-v2/out_35.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70002/FFB821CC-D83E-854D-9CD9-E676E41217D7.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018C-UL2018_MiniAODv2-v2/out_34.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/BBAA7C04-E2F7-4448-81FD-2B203AFCFBE5.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018C-UL2018_MiniAODv2-v2/out_641.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/130000/0E7C7C1B-6C0E-A144-8925-8D9677D8F7B6.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018C-UL2018_MiniAODv2-v2/out_1240.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/130000/A8AAEBA9-2C1B-D24F-A069-074FC3E47AA4.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018C-UL2018_MiniAODv2-v2/out_1243.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/DE7D6688-6AA2-0D48-9C8C-215B6CE8DC74.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018C-UL2018_MiniAODv2-v2/out_642.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/6BB3F0E9-88AE-6E44-AE74-C8E28BA063AD.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018C-UL2018_MiniAODv2-v2/out_645.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/5648C75C-EAF6-254B-B6F0-64D53F5FACCF.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018C-UL2018_MiniAODv2-v2/out_644.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/400D3817-91FF-514A-A18F-4B111A580882.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018C-UL2018_MiniAODv2-v2/out_438.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/07F5279E-6B94-5B40-86F7-E5C1C66BD159.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018C-UL2018_MiniAODv2-v2/out_439.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/5091B2EE-2F34-B442-8DCB-3035761C0545.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018C-UL2018_MiniAODv2-v2/out_436.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/1BA89478-F16C-6A49-915C-87967BCE9702.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018C-UL2018_MiniAODv2-v2/out_437.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/BD33D3C6-B1A8-9B43-9540-5B66D1B63218.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018C-UL2018_MiniAODv2-v2/out_434.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/544CB608-CFBD-1246-BE27-64E9BA25EE14.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018C-UL2018_MiniAODv2-v2/out_435.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/584F4F6F-F48D-1A47-8CDC-2DF960ECEAB3.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018C-UL2018_MiniAODv2-v2/out_432.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/AD137A51-8C0A-504C-AE72-441D620F8613.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018C-UL2018_MiniAODv2-v2/out_433.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/8B3A1167-6E33-1D45-8ECF-79C2F4BC1BF3.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018C-UL2018_MiniAODv2-v2/out_430.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/1CCDAA70-9CFB-6E44-ACF4-69074895527E.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018C-UL2018_MiniAODv2-v2/out_431.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/59DBA971-5C4C-4B48-A6CD-2D50238AEAE3.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018C-UL2018_MiniAODv2-v2/out_826.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/2A8A03BC-4EAE-064C-B6D5-A9A86358CAAE.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018C-UL2018_MiniAODv2-v2/out_339.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/C8A1D7A5-CCDB-CF49-9F4C-6BAC658A1234.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018C-UL2018_MiniAODv2-v2/out_338.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/12807786-F99C-AD4E-B36B-39E878845984.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018C-UL2018_MiniAODv2-v2/out_335.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/48678FB9-E6F6-7549-81AA-D3046898296F.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018C-UL2018_MiniAODv2-v2/out_334.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/EDCAECC4-64CA-4B4F-9F4C-6F0BE1BCEAA2.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018C-UL2018_MiniAODv2-v2/out_337.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/EFDE1DA3-A9F5-A544-8412-042734F70360.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018C-UL2018_MiniAODv2-v2/out_336.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/3907247D-207B-C64F-BEE8-080E5E72D8D1.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018C-UL2018_MiniAODv2-v2/out_331.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/3F2A2292-0350-8240-9A5A-14CB6831A5ED.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018C-UL2018_MiniAODv2-v2/out_330.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/B270FE51-0222-F343-ABCB-A83E6E8AA7CC.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018C-UL2018_MiniAODv2-v2/out_333.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/4243FAC4-5737-124D-925B-513ACCB38C3E.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018C-UL2018_MiniAODv2-v2/out_332.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70001/827C4150-B9EA-A34E-93EA-F4C029E70C5F.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018C-UL2018_MiniAODv2-v2/out_559.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/F0104EB6-325C-B74C-AD64-EFD937514DF3.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018C-UL2018_MiniAODv2-v2/out_1026.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70001/C2A844ED-ABB3-6F44-A1F6-F893FB4C8DD8.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018C-UL2018_MiniAODv2-v2/out_745.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/697400D3-775C-2D42-A026-2BD65D71AC5D.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018C-UL2018_MiniAODv2-v2/out_854.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/CF828601-F075-4544-856F-01FCE4AAFA94.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018C-UL2018_MiniAODv2-v2/out_818.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/9CE03475-9F61-8341-94C7-B90611DD074D.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018C-UL2018_MiniAODv2-v2/out_856.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/D9750F34-4CF8-9A4D-A63E-594679B2974F.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018C-UL2018_MiniAODv2-v2/out_857.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/94C0FF83-F552-FF4A-805D-B9A6BA9C288F.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018C-UL2018_MiniAODv2-v2/out_850.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/24E44019-0A66-9642-858A-0803E765A0A8.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018C-UL2018_MiniAODv2-v2/out_851.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/CC7B9544-704C-9543-917D-197494F6103B.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018C-UL2018_MiniAODv2-v2/out_852.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/16A5DA3A-5277-504C-AC47-573B581E3E82.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018C-UL2018_MiniAODv2-v2/out_0.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/5E294A17-20F6-F145-A6EC-A6039D94561B.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018C-UL2018_MiniAODv2-v2/out_858.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/4D532D26-31EC-0748-87CD-97DEBA6EBE78.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018C-UL2018_MiniAODv2-v2/out_859.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/87FD0F8B-A664-E64F-ADE3-4FB389EF6515.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018C-UL2018_MiniAODv2-v2/out_316.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70001/2180474B-7537-7B42-97B8-A8E316D6670E.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018C-UL2018_MiniAODv2-v2/out_748.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/EAA307EF-38AB-CE46-BEBB-6D5AAF5A39DC.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018C-UL2018_MiniAODv2-v2/out_6.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/098CD6E4-035F-D14B-9100-B03A1B036232.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018C-UL2018_MiniAODv2-v2/out_151.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70001/7FEF8234-4437-664C-AACB-D4941B03A623.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018C-UL2018_MiniAODv2-v2/out_425.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/E64AF03D-3B08-6541-A5CC-9F86231C339A.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018C-UL2018_MiniAODv2-v2/out_634.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/B605DB05-742F-2E4E-B8C5-BACEDA235A56.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018C-UL2018_MiniAODv2-v2/out_43.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/DE2F0091-0984-1E49-A802-2BB20F3F63A0.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018C-UL2018_MiniAODv2-v2/out_99.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/41A27DA1-DBF2-9947-B46F-9ABCFDF52AA7.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018C-UL2018_MiniAODv2-v2/out_98.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/26DC7867-D13A-8D40-9E8C-564053E2C80A.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018C-UL2018_MiniAODv2-v2/out_8.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/5FE0FEA1-AEB7-B842-89A3-C75D99B8B827.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018C-UL2018_MiniAODv2-v2/out_91.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70002/F1D023F3-E8E8-3C43-91FA-753AC9F0C1BF.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018C-UL2018_MiniAODv2-v2/out_90.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/7290E5A0-004F-0C49-9B46-975D91AB6972.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018C-UL2018_MiniAODv2-v2/out_93.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/DF96F62D-78A0-3D4A-9AFD-4244C84E19E8.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018C-UL2018_MiniAODv2-v2/out_92.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/C0E211EF-46CA-2F4E-9382-CCD94436817E.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018C-UL2018_MiniAODv2-v2/out_95.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/C5EDEAC2-CDC7-114F-A218-9B81E608229D.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018C-UL2018_MiniAODv2-v2/out_94.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/786FE2C4-7CC4-5F46-8A93-A53312EC3120.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018C-UL2018_MiniAODv2-v2/out_97.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/99FE625F-525D-D04A-BA54-3F81F548C773.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018C-UL2018_MiniAODv2-v2/out_96.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/19C90B77-36C5-D24D-959C-7ECDFBFABC7A.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018C-UL2018_MiniAODv2-v2/out_150.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/F2C7029D-19ED-3A40-8BD3-1F448CBDD4A0.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018C-UL2018_MiniAODv2-v2/out_424.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/FD135EB9-706D-444D-8CAA-5C72CA309C61.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018C-UL2018_MiniAODv2-v2/out_814.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/D892BD71-8071-5442-B82F-FA4C140CC189.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018C-UL2018_MiniAODv2-v2/out_314.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/EF706DFD-0095-CA41-A2BF-D4B2CDC02C65.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018C-UL2018_MiniAODv2-v2/out_815.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/F5A4BDB8-BAAF-C745-8640-E602FEAAFFE9.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018C-UL2018_MiniAODv2-v2/out_153.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/C777EF52-AE41-9142-8514-1A6781EC8B93.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018C-UL2018_MiniAODv2-v2/out_740.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/2659032D-B488-6C4C-8567-4108ADC84275.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018C-UL2018_MiniAODv2-v2/out_741.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/ED6B6FDC-145F-564C-8A9E-46B7739444F4.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018C-UL2018_MiniAODv2-v2/out_742.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70001/E7E56DBC-6650-534E-B3FA-E62D4042F0CD.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018C-UL2018_MiniAODv2-v2/out_743.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70001/FCBA79D9-27E7-FF41-A14D-410622078261.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018C-UL2018_MiniAODv2-v2/out_744.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70001/26F90EF1-3584-064C-9CD5-5D9D0FAC4A0E.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018C-UL2018_MiniAODv2-v2/out_558.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70001/40BCF305-2E5C-7D4A-BBB0-D9A82B4565EF.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018C-UL2018_MiniAODv2-v2/out_746.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70001/21EFBA5A-526A-5946-AB1B-02E1B66A8F90.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018C-UL2018_MiniAODv2-v2/out_747.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/1FD19FD8-B72E-8142-9951-2D757C729A5F.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018C-UL2018_MiniAODv2-v2/out_555.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70001/B2B78BF4-5992-134E-8476-7183ACB77F5C.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018C-UL2018_MiniAODv2-v2/out_554.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70001/F0D84D30-5BBF-A147-BB35-19D676F96DFC.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018C-UL2018_MiniAODv2-v2/out_557.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/5666AB54-00C3-C14D-8D8F-389891250EAE.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018C-UL2018_MiniAODv2-v2/out_556.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/E88E1492-B9A4-F54E-B73E-7CA24E428063.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018C-UL2018_MiniAODv2-v2/out_551.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/08CF9265-8004-E54C-8B41-254DD8EC8BF9.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018C-UL2018_MiniAODv2-v2/out_550.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/B9BE4083-0176-2542-9A53-512C7B229FE5.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018C-UL2018_MiniAODv2-v2/out_553.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/90F90AFA-83F2-EB4B-9599-60815E79017B.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018C-UL2018_MiniAODv2-v2/out_552.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/5E095C5A-C90E-F54D-8712-9F9D928CADAB.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018C-UL2018_MiniAODv2-v2/out_238.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/1A66A2D8-14CB-5942-8519-D6E6F90BA6DA.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018C-UL2018_MiniAODv2-v2/out_239.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/061DBDDC-41F5-2D4E-A287-C36F664C873E.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018C-UL2018_MiniAODv2-v2/out_234.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/25767C45-CFF1-4741-BD5A-68D3B6FC1578.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018C-UL2018_MiniAODv2-v2/out_235.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/7E79CE03-2C43-AE43-A0C4-C86137ED6AD6.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018C-UL2018_MiniAODv2-v2/out_236.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/9BECFDC2-5A3B-424E-9C9B-D16BD2EF468B.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018C-UL2018_MiniAODv2-v2/out_237.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/76D38721-F9BD-E84B-AE90-C9B70788F514.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018C-UL2018_MiniAODv2-v2/out_230.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/7BAAB9F4-5EFD-324B-98A9-459C915018E5.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018C-UL2018_MiniAODv2-v2/out_231.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/6558808A-4AD8-FD41-8D27-A17AE975BEF3.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018C-UL2018_MiniAODv2-v2/out_232.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/61E8F625-AA8E-4F43-B884-C77E57146FE3.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018C-UL2018_MiniAODv2-v2/out_233.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70001/CA6CFB72-F061-A244-B78C-763EE628702F.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018C-UL2018_MiniAODv2-v2/out_1050.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70001/F7ABF260-F482-7441-8649-B7FD10DEE929.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018C-UL2018_MiniAODv2-v2/out_1051.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70001/4480D8E4-FF86-0F4B-BDC2-562329C47C98.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018C-UL2018_MiniAODv2-v2/out_1052.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/87E1BE67-2157-EA4C-A5E7-BEE475F97704.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018C-UL2018_MiniAODv2-v2/out_1053.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/62EC28EB-F7D1-4D44-953E-004B59E25CA2.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018C-UL2018_MiniAODv2-v2/out_1054.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/ED2CD21E-BBAF-9D4E-8E63-53629412493B.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018C-UL2018_MiniAODv2-v2/out_1055.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/8E0087FC-37D7-BE45-BA79-4DBBF1AB4720.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018C-UL2018_MiniAODv2-v2/out_1056.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/B6F5A1DD-C734-D14F-A5D9-76E09F88B159.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018C-UL2018_MiniAODv2-v2/out_1057.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70001/A5859FE7-ADE3-7444-982B-7B29143DE368.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018C-UL2018_MiniAODv2-v2/out_1058.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70001/CBDB18EA-F361-C548-B27E-330E7168F0E5.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018C-UL2018_MiniAODv2-v2/out_1059.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70001/923535D6-F7A5-D047-B7F2-79D034FF3C3F.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018C-UL2018_MiniAODv2-v2/out_1193.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/330A6164-93F1-9847-B872-E3ED2DDA995E.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018C-UL2018_MiniAODv2-v2/out_1192.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/EE79DDFB-983A-7E42-BC06-EC6EF2ABFCDE.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018C-UL2018_MiniAODv2-v2/out_1195.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/91BDE5FA-EF43-3449-8767-2E9F4CD360C4.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018C-UL2018_MiniAODv2-v2/out_1194.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/0BD0EBE3-1958-C145-947C-72DCE7D6DCB5.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018C-UL2018_MiniAODv2-v2/out_1197.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/7BE0C2E0-DF40-D44B-BAE5-60382C1A58BB.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018C-UL2018_MiniAODv2-v2/out_1196.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/0743CC82-1799-234D-89C3-5DB4F468CFF9.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018C-UL2018_MiniAODv2-v2/out_1.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/9D1FC3F8-E183-5D49-BCA6-2F32771D87C1.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018C-UL2018_MiniAODv2-v2/out_1159.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/22CCC989-02C7-BF48-8F0F-9F7A2363550E.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018C-UL2018_MiniAODv2-v2/out_155.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/1428E057-E01A-C94C-BA0D-768766AEDAD0.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018C-UL2018_MiniAODv2-v2/out_159.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/21D17CCA-7CD1-EC4A-8E28-F1ACEFB07813.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018C-UL2018_MiniAODv2-v2/out_146.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/C9422CDA-3734-3840-A9F0-0F6D19497EA0.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018C-UL2018_MiniAODv2-v2/out_147.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/FC89E379-1571-5540-8E8E-D81F28B82562.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018C-UL2018_MiniAODv2-v2/out_144.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/D3FF5F4D-2E2D-1F42-BB0D-1BFFEC3EE14C.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018C-UL2018_MiniAODv2-v2/out_145.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/1AAE9FEE-0891-AF49-82C3-557CDD57444A.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018C-UL2018_MiniAODv2-v2/out_142.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/FA27FE2E-1F69-1F43-B9D5-04A9661C44D1.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018C-UL2018_MiniAODv2-v2/out_143.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/337BC149-5E30-7D42-B662-1CCC5A11A364.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018C-UL2018_MiniAODv2-v2/out_140.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/282E432A-430C-104A-BB48-9D033AC8AF4E.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018C-UL2018_MiniAODv2-v2/out_141.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/D3B26EEA-2EC3-3748-A620-ADE7B87FF968.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018C-UL2018_MiniAODv2-v2/out_612.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/6F9134EC-4330-AD42-BD45-1AE293669AB2.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018C-UL2018_MiniAODv2-v2/out_613.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/5D9F4E39-8688-DA47-B755-EF752FD81559.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018C-UL2018_MiniAODv2-v2/out_610.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/9B7173C3-DF9F-9C49-8793-BBF1D54CF502.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018C-UL2018_MiniAODv2-v2/out_611.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/E77ABAA1-02F2-7544-9EF8-E5087F4A9A67.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018C-UL2018_MiniAODv2-v2/out_616.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/B63AE8AB-848E-594C-971C-9B922E052F23.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018C-UL2018_MiniAODv2-v2/out_617.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70001/6C431410-C14E-1F47-B00C-8681F2FCA46B.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018C-UL2018_MiniAODv2-v2/out_148.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/6AAAEAD3-FA19-C446-B28E-678AA3A23CF2.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018C-UL2018_MiniAODv2-v2/out_149.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/CFB0F5AF-8314-B74C-BD18-D9D054F34FA1.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018C-UL2018_MiniAODv2-v2/out_511.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/727EBEB9-33DA-1B40-9BCF-0AA79B85C228.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018C-UL2018_MiniAODv2-v2/out_615.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/45BE567A-0E17-0242-AEFC-2F86DD25326D.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018C-UL2018_MiniAODv2-v2/out_510.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/CB4EE85C-198A-FC49-8552-CCB2F6C877F5.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018C-UL2018_MiniAODv2-v2/out_154.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/CF9A65D3-5363-8F47-B832-845182B9BA38.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018C-UL2018_MiniAODv2-v2/out_513.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/5B54B739-342D-8C4C-8DF2-491A0BD587AB.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018C-UL2018_MiniAODv2-v2/out_512.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/58ABC9D8-86F2-B24B-B818-C5D49334709E.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018C-UL2018_MiniAODv2-v2/out_948.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/23B579DB-B050-A04C-8C2C-AE1F5648B858.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018C-UL2018_MiniAODv2-v2/out_949.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/75A18580-5701-EC4B-900F-31BBE28476F8.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018C-UL2018_MiniAODv2-v2/out_946.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/04A2094C-C465-2A4F-A266-8085CE8CA9FE.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018C-UL2018_MiniAODv2-v2/out_947.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/35521155-2C47-054C-86F4-B774EB365395.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018C-UL2018_MiniAODv2-v2/out_944.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/41AED4CE-FA93-C545-8BBF-15772C444FD6.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018C-UL2018_MiniAODv2-v2/out_945.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/CE045EA0-4B51-8949-8BD0-5ED35BFF9577.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018C-UL2018_MiniAODv2-v2/out_942.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/EF3B6401-4789-0F47-98A4-C4416B3C7C71.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018C-UL2018_MiniAODv2-v2/out_943.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/184A7F3D-76A4-4D41-BBAF-C8DF8C9D80F5.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018C-UL2018_MiniAODv2-v2/out_940.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/AD5C187D-2D8C-284B-ADB8-7934D20B2676.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018C-UL2018_MiniAODv2-v2/out_941.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/049FC9F3-88A6-424E-ABD6-21A29A6E211C.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018C-UL2018_MiniAODv2-v2/out_1191.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/64DD32C1-6E40-5348-BFBA-B3120F342B81.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018C-UL2018_MiniAODv2-v2/out_345.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/41CEB324-6E2C-7C4D-B7C6-684AD6A6D08D.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018C-UL2018_MiniAODv2-v2/out_689.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/1B5EEDE9-F374-7E47-B1A7-F1739CA562A8.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018C-UL2018_MiniAODv2-v2/out_688.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70001/0FDA3A87-E0BB-D442-AA2B-0532261F73EC.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018C-UL2018_MiniAODv2-v2/out_157.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/811D70F4-3846-124A-81C7-7D7F3C096B8E.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018C-UL2018_MiniAODv2-v2/out_1190.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/AEE50651-8E63-C94C-A3CE-509190CF4474.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018C-UL2018_MiniAODv2-v2/out_685.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/70628175-8271-764C-A302-DEE31827C180.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018C-UL2018_MiniAODv2-v2/out_684.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/23BDAA6D-F01D-DE4B-AD38-0D3714A6754E.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018C-UL2018_MiniAODv2-v2/out_687.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/883B79E9-E367-AB40-853E-883774C36578.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018C-UL2018_MiniAODv2-v2/out_686.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/5E394751-069B-0D47-9444-973E21D078AE.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018C-UL2018_MiniAODv2-v2/out_681.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/31E27DCE-EA78-0143-ABA1-0F1690A4675B.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018C-UL2018_MiniAODv2-v2/out_680.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/A7171117-E326-E14D-9A05-043B9B725D89.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018C-UL2018_MiniAODv2-v2/out_683.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/E00767A9-9320-974D-B908-685615E304DF.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018C-UL2018_MiniAODv2-v2/out_682.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/78AEC0F3-4464-D248-8053-20ECA06DC8CF.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018C-UL2018_MiniAODv2-v2/out_458.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/4A883438-47F3-EA48-BBF1-A40C03B68036.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018C-UL2018_MiniAODv2-v2/out_459.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/259FFD4A-FFF3-D84E-A568-08920A422C16.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018C-UL2018_MiniAODv2-v2/out_621.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/6F09DE73-8B7D-4640-A1D9-AE3FFC18FD8F.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018C-UL2018_MiniAODv2-v2/out_873.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/43BAB6EB-2C61-0D4B-959F-941D7CC81B77.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018C-UL2018_MiniAODv2-v2/out_133.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/C6EB1215-BDB4-E94D-A3E7-230E9A456EA9.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018C-UL2018_MiniAODv2-v2/out_132.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/75CE88B6-DE71-7A40-8061-7F7AC5BBEDB7.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018C-UL2018_MiniAODv2-v2/out_131.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70001/ED5F276E-E8C0-DC48-901D-C112A350D9D8.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018C-UL2018_MiniAODv2-v2/out_130.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/33251A08-7ED3-D94C-A2F1-9EB4939206B3.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018C-UL2018_MiniAODv2-v2/out_137.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/47F8AA38-6B3F-444D-88D2-061FB3863FB3.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018C-UL2018_MiniAODv2-v2/out_136.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/79B340E0-0311-2C45-97DF-59E4F38EF395.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018C-UL2018_MiniAODv2-v2/out_135.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/BA681111-F627-394F-9464-5AC9A52BF2C4.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018C-UL2018_MiniAODv2-v2/out_134.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/65237BDC-F4A2-744C-9D30-81F38157A374.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018C-UL2018_MiniAODv2-v2/out_494.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/E269D386-32F2-BF4E-BA06-77C3CB5DAA51.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018C-UL2018_MiniAODv2-v2/out_495.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/6F3E5E86-7A80-7B4D-8D55-4250F85806EA.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018C-UL2018_MiniAODv2-v2/out_139.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/21ED4557-751A-7F42-A596-00E36C5290A8.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018C-UL2018_MiniAODv2-v2/out_138.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/D369EC8D-F643-664E-BFCE-6047AE31BE2D.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018C-UL2018_MiniAODv2-v2/out_490.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/4C063E21-0459-6E41-B013-392C62CBD5E6.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018C-UL2018_MiniAODv2-v2/out_491.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/EABA2822-5448-4B44-B642-DB5E55F6DD78.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018C-UL2018_MiniAODv2-v2/out_492.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/7AAF2A0C-551E-2F41-A928-00287209BB58.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018C-UL2018_MiniAODv2-v2/out_493.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/512297B1-C58B-8543-98F5-DE87318B6D07.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018C-UL2018_MiniAODv2-v2/out_24.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/C37B1D73-99BC-F147-8B37-F6BB683CC790.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018C-UL2018_MiniAODv2-v2/out_25.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/C53BF283-B960-2F4A-BE4A-4942ED8BC382.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018C-UL2018_MiniAODv2-v2/out_26.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70002/97DE4E7F-C740-1D4C-A34E-1C6D1EC1B880.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018C-UL2018_MiniAODv2-v2/out_27.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/F0BD6559-13CF-BD40-87D0-EAB6F1739735.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018C-UL2018_MiniAODv2-v2/out_20.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/E539A03D-AD3D-6243-B2C5-6765C2F2F342.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018C-UL2018_MiniAODv2-v2/out_21.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/4B1D5642-C847-5B42-B821-8DCDAEF72429.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018C-UL2018_MiniAODv2-v2/out_22.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/AB249229-CFD5-6741-B1B7-25A5C6D2DB5D.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018C-UL2018_MiniAODv2-v2/out_23.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/93E7CFB0-5ACB-F347-8431-7406393421B3.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018C-UL2018_MiniAODv2-v2/out_1029.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/4B7F87D9-3D7C-FB42-8E38-01003796E126.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018C-UL2018_MiniAODv2-v2/out_1220.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/DAD6F8C5-929D-F141-A3E9-709898D0A7AB.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018C-UL2018_MiniAODv2-v2/out_28.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70002/548F715D-10B1-714E-8278-D680CCFAF82D.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018C-UL2018_MiniAODv2-v2/out_29.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/47A0FEDE-08F5-C741-A2EE-56D2763F8160.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018C-UL2018_MiniAODv2-v2/out_820.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/2F8F35A5-861F-254F-B826-12727D7E226F.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018C-UL2018_MiniAODv2-v2/out_407.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/EDCDD6B0-0D8D-2A46-8443-60F8372D0012.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018C-UL2018_MiniAODv2-v2/out_406.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/A056F095-71D3-2A47-98E5-D7686C51D661.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018C-UL2018_MiniAODv2-v2/out_405.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/C73FEC74-28EC-D948-8AE4-A6F970E45C49.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018C-UL2018_MiniAODv2-v2/out_404.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/9CB2F6B1-B034-5F43-B993-0B37D9005124.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018C-UL2018_MiniAODv2-v2/out_403.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/487D2889-5BF0-C94F-B064-89B5C9FD3174.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018C-UL2018_MiniAODv2-v2/out_402.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/6CEE66DE-E1CD-1F42-83DD-408D0DF8D032.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018C-UL2018_MiniAODv2-v2/out_401.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/95A59BE9-D8AA-3643-81EA-DA04C24BFE3E.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018C-UL2018_MiniAODv2-v2/out_400.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/87D92C9E-3B45-BE47-B7B0-0A77ACE5A215.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018C-UL2018_MiniAODv2-v2/out_933.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/14331FAD-D98B-D942-82E9-4275DBB489F5.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018C-UL2018_MiniAODv2-v2/out_932.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70001/550819B4-A35A-D047-A2AE-E782BD7361C5.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018C-UL2018_MiniAODv2-v2/out_931.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/CC1019C7-809D-3F46-91E8-BD69F6285CD2.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018C-UL2018_MiniAODv2-v2/out_930.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/F538903E-31CC-3A4D-9C75-5E78BCAC4B88.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018C-UL2018_MiniAODv2-v2/out_937.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/D6489B29-D60E-C947-A1C6-590E3929EFEA.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018C-UL2018_MiniAODv2-v2/out_452.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/D81700F9-A38A-D24C-A686-3BC7D6C2DB66.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018C-UL2018_MiniAODv2-v2/out_409.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/DCEC6034-6DC7-4447-9038-635BB450C57D.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018C-UL2018_MiniAODv2-v2/out_408.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/350D0218-02AF-7541-B7D2-3779EB72CC83.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018C-UL2018_MiniAODv2-v2/out_453.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/7B37AB31-150F-214E-81BB-93FFA50A1453.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018C-UL2018_MiniAODv2-v2/out_879.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/3017473D-0E39-A344-B610-A29532A73251.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018C-UL2018_MiniAODv2-v2/out_414.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/8B2090CD-F564-D845-B515-DDEBEE5E5405.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018C-UL2018_MiniAODv2-v2/out_455.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/B2D33762-590C-C446-AD1F-B286625C548F.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018C-UL2018_MiniAODv2-v2/out_456.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70001/DE414488-C0F7-8A43-B3B1-F40B3DB23D8A.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018C-UL2018_MiniAODv2-v2/out_1118.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/10E241C6-766D-194A-8113-B8D4974DD4F9.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018C-UL2018_MiniAODv2-v2/out_457.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/89A2E6C6-FB67-8140-82D7-F5A71008388B.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018C-UL2018_MiniAODv2-v2/out_183.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/4AE84480-8296-B74E-A69B-08E83135F5DD.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018C-UL2018_MiniAODv2-v2/out_849.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/80FB962B-5773-E147-ABD8-1935364BC367.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018C-UL2018_MiniAODv2-v2/out_415.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70001/3EA290BA-9A05-F347-B547-E9EB0F03950A.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018C-UL2018_MiniAODv2-v2/out_737.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70002/32057B72-8ABE-504D-9DE0-C1967C361272.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018C-UL2018_MiniAODv2-v2/out_379.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70002/AE323FC0-082C-EB4C-92B1-A2EBBA3A1199.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018C-UL2018_MiniAODv2-v2/out_378.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/E9D89A07-C2FA-5C41-80C9-34453BB9B9A3.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018C-UL2018_MiniAODv2-v2/out_228.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/D55317EE-9084-6C46-A527-61061174DC5C.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018C-UL2018_MiniAODv2-v2/out_829.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/892BC780-45C2-A047-A9B7-DB72E888F14A.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018C-UL2018_MiniAODv2-v2/out_828.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/22EAFEF3-6A24-3F46-A94A-AB28297AA7E5.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018C-UL2018_MiniAODv2-v2/out_44.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/F7C05B3E-FC06-3E4A-A225-FE92DCCDD688.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018C-UL2018_MiniAODv2-v2/out_371.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/4A9D1B8C-4CD0-1040-9B45-E0D5956B68AC.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018C-UL2018_MiniAODv2-v2/out_370.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/090E4878-CE53-B541-AA67-870484CEB86A.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018C-UL2018_MiniAODv2-v2/out_373.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/D69FB550-B801-8246-87BA-B52FCC9B6FE5.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018C-UL2018_MiniAODv2-v2/out_372.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/6F8ADD5A-1D1E-EF49-BF94-DDD1F1B2567B.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018C-UL2018_MiniAODv2-v2/out_375.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/B344821B-BC0F-8B45-9E49-71BADD30B51F.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018C-UL2018_MiniAODv2-v2/out_374.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/1B88C0A2-2354-5A4C-8CDD-B85ACFCE806A.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018C-UL2018_MiniAODv2-v2/out_377.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/E8D185C6-FA74-F34D-B1F5-3E074EBA627E.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018C-UL2018_MiniAODv2-v2/out_376.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/869345DD-C92D-0548-BD14-1007EBD320D4.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018C-UL2018_MiniAODv2-v2/out_1177.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/074258BC-16CE-0A46-AA57-A7D9CB6A2D94.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018C-UL2018_MiniAODv2-v2/out_708.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/26DD1602-FD68-FE43-8650-E51920274099.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018C-UL2018_MiniAODv2-v2/out_709.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70001/E4C445F9-0297-B542-9DE2-7F2AD6F816A0.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018C-UL2018_MiniAODv2-v2/out_1176.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/BA1863CB-EA23-3642-8751-8EB476926A2D.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018C-UL2018_MiniAODv2-v2/out_704.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/8773F8A3-B592-4A4B-A4F3-917A13E80548.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018C-UL2018_MiniAODv2-v2/out_705.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/260EF349-AD8B-B046-8663-A83FEA78AB0C.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018C-UL2018_MiniAODv2-v2/out_706.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/245A77BC-0DF3-3045-9EAE-BE5212F6F902.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018C-UL2018_MiniAODv2-v2/out_707.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/29BB2588-F1A9-A44D-935D-F39FF0A5D622.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018C-UL2018_MiniAODv2-v2/out_700.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70001/991764BE-B358-DA42-B6D4-2A5FDB98E996.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018C-UL2018_MiniAODv2-v2/out_1175.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/A8A7E149-7B9A-F140-B2D2-70FDF1CEDE84.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018C-UL2018_MiniAODv2-v2/out_702.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/7A3AEC39-26CD-BE4A-B079-9713AA315ACA.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018C-UL2018_MiniAODv2-v2/out_703.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/E7BBF3E9-402E-2A43-BADF-329B4D68A47E.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018C-UL2018_MiniAODv2-v2/out_393.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/A442C2D6-F4E1-3444-9F64-7C6E414FA24F.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018C-UL2018_MiniAODv2-v2/out_392.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/5A69B176-2165-894A-889F-051DAD084637.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018C-UL2018_MiniAODv2-v2/out_88.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/7047402A-3060-934F-92E7-36665C0667AD.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018C-UL2018_MiniAODv2-v2/out_619.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70002/10173EC9-96E2-8645-A607-EC1C94756A1B.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018C-UL2018_MiniAODv2-v2/out_397.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/562C18B7-C2C6-3948-8BD0-3DF878F85ECF.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018C-UL2018_MiniAODv2-v2/out_396.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/2730EE28-FA1F-114B-892B-1E6FF6D5CFDE.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018C-UL2018_MiniAODv2-v2/out_395.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/DCF1C387-C91C-B54D-89E0-A904120C0A07.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018C-UL2018_MiniAODv2-v2/out_394.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/A0159E88-CF47-9A47-89ED-2181585DEAA2.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018C-UL2018_MiniAODv2-v2/out_82.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/B184B823-C4A1-9D4D-B1AB-0880F3C68CAE.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018C-UL2018_MiniAODv2-v2/out_83.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/F7C62A15-01E1-5941-9843-286D46429305.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018C-UL2018_MiniAODv2-v2/out_80.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/58F3E6F4-3384-504D-9DB2-E76C0F3E1653.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018C-UL2018_MiniAODv2-v2/out_81.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70001/515FB3B5-DBD1-BA4A-96BC-0E45A5B404D6.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018C-UL2018_MiniAODv2-v2/out_86.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/26D5A539-FBF1-C04A-AF62-3D7797C9C237.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018C-UL2018_MiniAODv2-v2/out_87.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/70ABD40B-1ADC-AE49-B218-B9D0E71B8DDC.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018C-UL2018_MiniAODv2-v2/out_84.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70001/34B8644A-70D2-1A4E-9DB9-6373FD370438.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018C-UL2018_MiniAODv2-v2/out_85.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/F682573E-BA67-234D-AE54-DF6271F3B6BC.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018C-UL2018_MiniAODv2-v2/out_797.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/B37DB0CD-0F64-0742-A2D1-F9BF1F80A5C6.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018C-UL2018_MiniAODv2-v2/out_796.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/35EF7F9B-AE33-7848-A316-D7DBAE869076.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018C-UL2018_MiniAODv2-v2/out_795.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/5D1D9FEE-1D2E-BC4E-8B4E-BC05C95A3F38.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018C-UL2018_MiniAODv2-v2/out_794.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/3BD863FE-561C-1D4A-A7D3-6D79F4879240.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018C-UL2018_MiniAODv2-v2/out_793.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70001/46EFBFA4-365B-1B44-A673-9212463AD1C3.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018C-UL2018_MiniAODv2-v2/out_792.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70001/7A315BEB-080B-074C-B9A9-0366BAF8B1DB.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018C-UL2018_MiniAODv2-v2/out_791.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/789557F9-491E-524D-B216-20F6BFAE9D00.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018C-UL2018_MiniAODv2-v2/out_790.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70001/82792785-8A36-F248-820B-7FD4BA6B57E4.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018C-UL2018_MiniAODv2-v2/out_1170.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70001/B42F9052-3EF8-874C-8DA2-0BC2A324DBBF.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018C-UL2018_MiniAODv2-v2/out_749.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/41945CFD-DFFF-FF47-BD56-A9FF8FFA1064.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018C-UL2018_MiniAODv2-v2/out_799.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/CA75B9C5-C768-C644-BFE4-C79F9322F2C1.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018C-UL2018_MiniAODv2-v2/out_798.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/8641C0FC-0EFB-0D48-B25C-36B20C2FE6E4.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018C-UL2018_MiniAODv2-v2/out_7.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/3631A9A4-E0EC-F740-84AD-0E151F288623.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018C-UL2018_MiniAODv2-v2/out_170.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/2FBD8EFE-048A-D745-B41B-501ACB9F3C84.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018C-UL2018_MiniAODv2-v2/out_586.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/DA280F4A-E002-BE40-AE41-1DBC7CA2F464.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018C-UL2018_MiniAODv2-v2/out_587.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/A4BC7620-2C5A-5343-93D7-AA2772FEF94C.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018C-UL2018_MiniAODv2-v2/out_1139.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/63547594-ADCA-1040-903D-512AA633C791.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018C-UL2018_MiniAODv2-v2/out_585.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/F4021FB1-8B16-5447-82AD-83746553B278.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018C-UL2018_MiniAODv2-v2/out_582.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/7C8F0594-3B9A-7240-982B-1F42EE129F3A.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018C-UL2018_MiniAODv2-v2/out_583.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/3197AF53-3F97-1440-8D07-3C4154A0C0C1.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018C-UL2018_MiniAODv2-v2/out_580.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/0DA3AF83-F98C-1049-8E09-886448C06F71.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018C-UL2018_MiniAODv2-v2/out_581.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/544AE516-0076-3E49-AB40-4D944CACB254.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018C-UL2018_MiniAODv2-v2/out_1133.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/0B742601-C236-F846-800C-8E7F2FBAECF8.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018C-UL2018_MiniAODv2-v2/out_1132.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/5C7114D2-05C1-D04F-A95A-65DE6FBF8EA5.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018C-UL2018_MiniAODv2-v2/out_1131.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/49664D85-82BA-0342-A114-4454F14F9582.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018C-UL2018_MiniAODv2-v2/out_1130.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/593044AB-C355-A340-8345-85EB0366239B.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018C-UL2018_MiniAODv2-v2/out_1137.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/7D347434-2959-7B4D-B2F1-73D781C6DF29.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018C-UL2018_MiniAODv2-v2/out_1136.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/435C54DC-343D-664D-A8C0-A660A9D348C1.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018C-UL2018_MiniAODv2-v2/out_1135.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/83139626-D4E2-1540-80D0-F31DD7B1B4B9.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018C-UL2018_MiniAODv2-v2/out_589.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70002/15AA7A0C-13AA-254F-9165-B0CBA4EDC479.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018C-UL2018_MiniAODv2-v2/out_245.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70002/BC7D3D2A-C5E6-6E44-9138-147CCE86A909.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018C-UL2018_MiniAODv2-v2/out_244.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/F320C1BD-7540-F545-B626-3FDB9B1DDCC5.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018C-UL2018_MiniAODv2-v2/out_247.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70002/154E67A9-3128-DD4D-BBCE-4424BA4D952C.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018C-UL2018_MiniAODv2-v2/out_246.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/E82923CD-0B1E-2B49-A106-093E63E661E2.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018C-UL2018_MiniAODv2-v2/out_241.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/43CB2736-6AF6-A343-8C03-406C14306770.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018C-UL2018_MiniAODv2-v2/out_614.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70002/F47D3D88-ED29-1244-8649-3040462F3454.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018C-UL2018_MiniAODv2-v2/out_243.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/1FA1C137-B554-4D4A-A1CC-E5EFEEF5DB32.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018C-UL2018_MiniAODv2-v2/out_242.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/CDF53555-9130-B940-BD26-47B263926A5B.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018C-UL2018_MiniAODv2-v2/out_1178.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70002/DA5A49E7-F33E-3745-B3CD-89C12C391AA4.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018C-UL2018_MiniAODv2-v2/out_249.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/7A1C923B-569A-2D45-8FFA-B4C07AFEF136.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018C-UL2018_MiniAODv2-v2/out_248.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/9A4A2BF2-3568-B74D-88A6-6DD9DF89414B.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018C-UL2018_MiniAODv2-v2/out_972.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/7340AB17-37C7-3147-BC1E-EE88EAA219ED.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018C-UL2018_MiniAODv2-v2/out_418.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/F501DC92-1A25-8A46-8EBA-995A66828E44.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018C-UL2018_MiniAODv2-v2/out_224.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/1CAE4B5F-B751-B546-B49F-22D3E13D7E28.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018C-UL2018_MiniAODv2-v2/out_639.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/5E7A0988-99F2-E74A-9F97-616DDF25E509.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018C-UL2018_MiniAODv2-v2/out_419.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/3F2A19C8-EB42-6E49-9408-F218192C9417.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018C-UL2018_MiniAODv2-v2/out_519.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/6E73FEB8-EDD1-5445-AB5E-8C32C711BC34.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018C-UL2018_MiniAODv2-v2/out_518.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/DACD3607-F86C-FD42-843E-B4B4E43C892E.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018C-UL2018_MiniAODv2-v2/out_1009.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/B32DDAB0-214F-254C-A39F-1D54BB5E22E4.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018C-UL2018_MiniAODv2-v2/out_1008.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/46BD9BF3-DA1B-EC4C-B911-ACDF8720198A.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018C-UL2018_MiniAODv2-v2/out_1007.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/92A27FED-C0A9-C449-A176-156A6023F894.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018C-UL2018_MiniAODv2-v2/out_1006.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/9FEDD350-517C-184C-8155-79E7C1E755FB.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018C-UL2018_MiniAODv2-v2/out_1005.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/5FAC4AD3-F18F-294A-94C9-548CE96C432A.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018C-UL2018_MiniAODv2-v2/out_1004.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/6F60A763-FCE6-8A49-8FA9-437759944973.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018C-UL2018_MiniAODv2-v2/out_515.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/3220CE51-98A7-CB46-BBD7-04E61C61CF11.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018C-UL2018_MiniAODv2-v2/out_1002.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/63EC50E0-7C22-BC4E-B914-8F956B3DD8F4.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018C-UL2018_MiniAODv2-v2/out_517.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/6D044455-302C-2F42-A24E-59F8DC3CBC98.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018C-UL2018_MiniAODv2-v2/out_1000.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/D13CC330-F7E4-2547-B54E-B309E22B7BE2.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018C-UL2018_MiniAODv2-v2/out_623.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/97C9AE9A-E776-474C-8744-8ADD16E97C3B.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018C-UL2018_MiniAODv2-v2/out_622.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/065708C5-19D2-B143-AE7E-E022B202C200.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018C-UL2018_MiniAODv2-v2/out_1225.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/CD9EA9F8-81BB-BC42-B51F-E73FB5CF9F8A.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018C-UL2018_MiniAODv2-v2/out_514.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/7BC4F67C-FF53-5D49-B060-B6B24D6F7E60.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018C-UL2018_MiniAODv2-v2/out_627.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/B5C126B5-C3B0-0741-BCE7-5AA730011592.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018C-UL2018_MiniAODv2-v2/out_626.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/3DCB8C63-44A4-DC41-8D03-D115BC37B3A7.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018C-UL2018_MiniAODv2-v2/out_625.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/B9DED82F-57E2-244C-8A1D-BF11DAB8CA64.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018C-UL2018_MiniAODv2-v2/out_624.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/D1E4889A-B727-6446-ADAD-271517898687.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018C-UL2018_MiniAODv2-v2/out_450.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/EFDBFA3E-66DC-254B-B819-153C34DDE652.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018C-UL2018_MiniAODv2-v2/out_451.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/6C84CB9E-0213-374F-B322-4CFC2E1125A5.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018C-UL2018_MiniAODv2-v2/out_629.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/FAD1ADCF-99D1-DF41-B85E-30A35E217BB0.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018C-UL2018_MiniAODv2-v2/out_628.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/10AF2381-A0C6-7343-B3B7-7281EEDEEDB3.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018C-UL2018_MiniAODv2-v2/out_454.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/FDBBBAB5-2B5F-AF4C-945C-ED94646943CF.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018C-UL2018_MiniAODv2-v2/out_1085.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70001/766B339F-984F-C64C-A2F0-76702CB3C325.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018C-UL2018_MiniAODv2-v2/out_1229.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/93A7551A-DB25-A845-AC1D-82FAEA26B681.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018C-UL2018_MiniAODv2-v2/out_1228.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/EAF8B9A8-65DA-D844-B415-08065A40D232.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018C-UL2018_MiniAODv2-v2/out_1084.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/6BE50CC9-9A8B-EB4E-8CF2-E92D0E5E6441.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018C-UL2018_MiniAODv2-v2/out_179.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/F5C1907B-DD06-424B-B64E-C10F42F62027.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018C-UL2018_MiniAODv2-v2/out_178.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/E433FC2C-413C-A74C-99FF-7891DC7AC590.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018C-UL2018_MiniAODv2-v2/out_177.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/F28FBC88-1D63-1345-B9C9-75D110DE3F88.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018C-UL2018_MiniAODv2-v2/out_199.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/80C2F1C2-4850-3E4F-B8A4-FB94768149F3.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018C-UL2018_MiniAODv2-v2/out_175.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/3D7F3600-B45C-7347-A7FA-377207B6D9F4.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018C-UL2018_MiniAODv2-v2/out_174.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/85E64B97-BE06-D047-BD65-4F8B483841A2.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018C-UL2018_MiniAODv2-v2/out_173.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/37B1A1CB-92E5-9449-91DE-61606F4A175E.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018C-UL2018_MiniAODv2-v2/out_172.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/17E37134-79FC-724D-8632-CCA4F90782F8.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018C-UL2018_MiniAODv2-v2/out_171.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/1A24EEF3-EF6D-B345-9D64-E5507E6217F9.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018C-UL2018_MiniAODv2-v2/out_594.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70001/A2709C21-9D8D-CF4F-B0B3-BC7F790BBA63.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018C-UL2018_MiniAODv2-v2/out_977.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/1201694E-B35E-234D-B2FF-22AB7B6BB5E5.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018C-UL2018_MiniAODv2-v2/out_182.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/13A3B240-BE2F-0F48-B8E1-897169BF554B.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018C-UL2018_MiniAODv2-v2/out_975.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/617E27B0-28D8-974F-9554-9574BA1B4F9D.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018C-UL2018_MiniAODv2-v2/out_974.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/00E52E1D-BB50-1547-B310-DD7EF7D54B81.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018C-UL2018_MiniAODv2-v2/out_973.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/143223C3-4C52-9147-95A5-B395092D6FFD.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018C-UL2018_MiniAODv2-v2/out_1081.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/DCBC63CA-27BC-B747-96AD-B93AB9D16940.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018C-UL2018_MiniAODv2-v2/out_971.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70001/1BB44F31-E4B6-954C-8820-4C654C343844.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018C-UL2018_MiniAODv2-v2/out_970.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/586F2DC8-8EC2-134D-9663-CB08BEE74F56.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018C-UL2018_MiniAODv2-v2/out_1080.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/090B7F2F-6BB9-864F-8B1D-786EE748D655.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018C-UL2018_MiniAODv2-v2/out_180.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/DD5B91B4-2585-2542-B0A1-81CC258D1AC4.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018C-UL2018_MiniAODv2-v2/out_979.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70001/2364F5B5-F0DA-EE48-BF61-F81BACC6E13C.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018C-UL2018_MiniAODv2-v2/out_978.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/B4F9E523-E8CB-FE49-994A-55D163279E59.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018C-UL2018_MiniAODv2-v2/out_656.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70001/CA32E383-B674-F749-B943-CC8BF385B27F.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018C-UL2018_MiniAODv2-v2/out_1120.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/1538057F-0EC0-CD4F-94BD-D8D31C385AD5.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018C-UL2018_MiniAODv2-v2/out_654.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/24705F90-67A3-6E4F-A720-5AC818D7C9B8.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018C-UL2018_MiniAODv2-v2/out_181.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/50EFC2D7-6D56-0044-A6E3-BA49B2AB4ED9.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018C-UL2018_MiniAODv2-v2/out_186.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/7FAEFEC3-0CC6-6D47-8414-792A861B1D71.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018C-UL2018_MiniAODv2-v2/out_187.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/EB1D1481-86CB-F445-976E-B766E74839A7.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018C-UL2018_MiniAODv2-v2/out_184.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/CEEEE363-C321-4E43-94D1-96986A919306.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018C-UL2018_MiniAODv2-v2/out_185.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/FDFF5B34-297E-8E4A-9A39-E4AC8F45DDE3.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018C-UL2018_MiniAODv2-v2/out_652.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/37B2293D-10BF-3845-BF55-1D01E95AD301.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018C-UL2018_MiniAODv2-v2/out_188.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/17BDA55C-0455-F348-B839-B8190F473CEA.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018C-UL2018_MiniAODv2-v2/out_189.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/0197E816-3420-924E-B99F-A15FAECAB281.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018C-UL2018_MiniAODv2-v2/out_1122.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/8DC1BB67-BEE4-A74B-974D-7779D5FD26AA.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018C-UL2018_MiniAODv2-v2/out_658.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/E39D04E1-81F1-3741-938D-1BC8CD2304B0.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018C-UL2018_MiniAODv2-v2/out_516.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/E8044666-E161-964B-BFF2-3D371A572589.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018C-UL2018_MiniAODv2-v2/out_1123.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/BD505F4B-B727-5646-99E8-4E63F0463CC0.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018C-UL2018_MiniAODv2-v2/out_650.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/F6B0AEEA-CB4C-6848-AFC2-78F4B06DCC92.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018C-UL2018_MiniAODv2-v2/out_1124.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/5B6DF7BB-8A26-5340-AF9F-78219FF22356.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018C-UL2018_MiniAODv2-v2/out_651.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/B19E0778-C7D0-D548-9E2E-0577B777D23E.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018C-UL2018_MiniAODv2-v2/out_1125.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/4CD03AD7-AFD5-3A43-B71B-C1597349D441.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018C-UL2018_MiniAODv2-v2/out_1089.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/786D4973-A51D-144C-9311-A27DCFE585E1.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018C-UL2018_MiniAODv2-v2/out_1088.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/2E28D554-ED74-F84E-BC6D-BE46C858DEC5.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018C-UL2018_MiniAODv2-v2/out_1227.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/7747DE14-0683-5F41-8975-EC78538CF522.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018C-UL2018_MiniAODv2-v2/out_11.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/0BFEFA41-614E-584C-9542-519DFBC636C4.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018C-UL2018_MiniAODv2-v2/out_10.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/BE69D61F-5ACB-C84E-A000-151322826E79.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018C-UL2018_MiniAODv2-v2/out_13.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/3CB40C58-5B50-B14A-A09F-78602C43927E.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018C-UL2018_MiniAODv2-v2/out_12.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/CC0A930E-DFE5-8444-93CC-212ED569E91B.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018C-UL2018_MiniAODv2-v2/out_15.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/9C3FB11B-7587-9B41-872A-4E85EC5AEB6E.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018C-UL2018_MiniAODv2-v2/out_14.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/7657A656-3070-A841-BE59-B76CE046E66A.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018C-UL2018_MiniAODv2-v2/out_17.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/4473AB06-130D-B045-852C-3D1D75F901D6.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018C-UL2018_MiniAODv2-v2/out_16.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/B0ADA59A-7737-864B-8EA6-BE74F934F949.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018C-UL2018_MiniAODv2-v2/out_19.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/4E66BA61-9D99-BA42-93D6-668328452F9B.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018C-UL2018_MiniAODv2-v2/out_18.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/4F2F2B2C-743E-7D4D-8E2B-AF69693C9CF1.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018C-UL2018_MiniAODv2-v2/out_863.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/0E736100-62AC-774E-9538-F5B838288593.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018C-UL2018_MiniAODv2-v2/out_862.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/AC4341AE-E182-BA4D-A2A3-AE8E57730B1F.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018C-UL2018_MiniAODv2-v2/out_865.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/E6C55D00-CFF5-3842-BF55-1B8E54DEB1B1.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018C-UL2018_MiniAODv2-v2/out_864.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/31E98022-F891-DB4B-BCD9-D9674C5F55B0.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018C-UL2018_MiniAODv2-v2/out_867.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/3802462B-FC53-D14D-BC48-4490EEE4E47F.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018C-UL2018_MiniAODv2-v2/out_866.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70002/67431A3E-180A-0449-BEB2-7DA2501C35E5.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018C-UL2018_MiniAODv2-v2/out_354.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70001/D9806A0E-DD1C-C841-9C3B-B4E219EA06F5.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018C-UL2018_MiniAODv2-v2/out_1210.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/9ABE7FF9-75E1-D845-87ED-1A3A904DAD24.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018C-UL2018_MiniAODv2-v2/out_659.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/A756DC78-4026-BA41-9729-5455346CCFF5.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018C-UL2018_MiniAODv2-v2/out_1226.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/2B32BB9C-96D0-6341-AE69-39143C222D5F.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018C-UL2018_MiniAODv2-v2/out_883.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/5622D265-7ED7-3D4A-8A4E-BF5DAEE0ADFC.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018C-UL2018_MiniAODv2-v2/out_882.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/08E58694-DA1E-B144-BE9B-50505B0358D8.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018C-UL2018_MiniAODv2-v2/out_881.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/F0977731-6295-7B4C-8DA5-D416CA9FAD3E.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018C-UL2018_MiniAODv2-v2/out_880.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/A73CB40C-A885-1D42-BFD9-23F3AF3C244F.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018C-UL2018_MiniAODv2-v2/out_887.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/7ADA2063-4BF8-3648-AA72-45312B583F26.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018C-UL2018_MiniAODv2-v2/out_886.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/4018D32B-B882-564C-AE69-FFBA18BBFCD3.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018C-UL2018_MiniAODv2-v2/out_885.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/9A1E3C16-FBC0-7E4B-8126-029106EE8579.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018C-UL2018_MiniAODv2-v2/out_884.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/C4A7A167-43E9-9E4D-9846-0A2E07455191.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018C-UL2018_MiniAODv2-v2/out_889.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/7B8B26C5-0A75-8A49-BFE3-1A43B1BFD55F.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018C-UL2018_MiniAODv2-v2/out_888.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70001/A9668D6B-5A90-4F47-958F-DFDD0473C835.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018C-UL2018_MiniAODv2-v2/out_116.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/6B52CC12-153A-BB46-AA22-E65D00BEB5FB.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018C-UL2018_MiniAODv2-v2/out_45.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/7136960A-9744-B545-9188-468B1411973B.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018C-UL2018_MiniAODv2-v2/out_657.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70002/5977B864-85A8-0C4B-968C-3B2CACB3EABC.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018C-UL2018_MiniAODv2-v2/out_356.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/2AD46575-21BA-D544-AB30-1EAB5B36D859.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018C-UL2018_MiniAODv2-v2/out_1083.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70002/3D6A125C-2879-B44F-9F4A-31AFF3BCBE4C.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018C-UL2018_MiniAODv2-v2/out_355.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70002/0DEE86D8-94AC-8F4F-9337-F5AD13ACC75D.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018C-UL2018_MiniAODv2-v2/out_322.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70002/F5C97603-9503-8B49-8460-C2A1C86421A8.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018C-UL2018_MiniAODv2-v2/out_323.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70002/14A6BE3C-5BE2-334E-884B-4E5ADC56822F.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018C-UL2018_MiniAODv2-v2/out_320.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70002/592985BA-D35A-D74F-BDDE-4AD67983E175.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018C-UL2018_MiniAODv2-v2/out_321.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/127C6C9A-9029-4849-A722-D2186AE59974.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018C-UL2018_MiniAODv2-v2/out_326.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/E0B3A07A-8CDA-C940-AEA5-9D2F1E182362.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018C-UL2018_MiniAODv2-v2/out_327.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70002/9B49619C-4BAF-D045-8B53-1A048B875FF4.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018C-UL2018_MiniAODv2-v2/out_324.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/45513272-4CDC-2F4B-ABDB-12FA98AD7912.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018C-UL2018_MiniAODv2-v2/out_325.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/4B4E56D3-B049-9C45-AD9E-0185E0AE7D99.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018C-UL2018_MiniAODv2-v2/out_1224.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/6885EA55-9349-AB45-A3A5-3321E0942B4D.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018C-UL2018_MiniAODv2-v2/out_328.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/0380516C-2708-3E4F-BE8B-03DBDF39A0B0.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018C-UL2018_MiniAODv2-v2/out_329.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70001/6D60EA9A-0F91-E64B-9542-96CE3E10217E.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018C-UL2018_MiniAODv2-v2/out_562.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/F46824B5-45D7-1044-B2A2-B35F16364C97.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018C-UL2018_MiniAODv2-v2/out_201.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/05A47626-1B39-DF44-9311-5EFA3FC88327.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018C-UL2018_MiniAODv2-v2/out_200.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/588568AB-DC52-1946-A3B1-1F922571F8E9.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018C-UL2018_MiniAODv2-v2/out_203.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/24B3DD7B-5C13-0E44-8DC7-E5C77E436BE8.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018C-UL2018_MiniAODv2-v2/out_202.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/3C753A87-439C-F341-B748-8AE0931A49D1.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018C-UL2018_MiniAODv2-v2/out_205.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/BD9855F8-B1F4-FB44-8ABA-E93BD9E4574A.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018C-UL2018_MiniAODv2-v2/out_204.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/8C81EFDF-0BA0-8E4F-98BB-C27A8EAB1C10.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018C-UL2018_MiniAODv2-v2/out_207.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/8965DFA6-19B8-4541-BC7E-650D7A93F3A0.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018C-UL2018_MiniAODv2-v2/out_206.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/483D3E81-F7FA-6748-ABA8-0E993646E182.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018C-UL2018_MiniAODv2-v2/out_209.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/2DF25DCD-0681-3546-997C-56DF5393E853.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018C-UL2018_MiniAODv2-v2/out_208.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/9CA4B0C3-7C06-7B4B-88D6-886ACD54C61A.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018C-UL2018_MiniAODv2-v2/out_779.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/0CB1C938-EFC8-2548-9AB6-F0171F77A97E.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018C-UL2018_MiniAODv2-v2/out_778.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/F9B3D4C7-04C1-8B46-A420-138C095C4DFF.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018C-UL2018_MiniAODv2-v2/out_679.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/8EAE2C81-F753-DD44-9D48-5881ABBE5884.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018C-UL2018_MiniAODv2-v2/out_77.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/FA52FA10-3619-E04B-90F9-2E12D7E4B7C6.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018C-UL2018_MiniAODv2-v2/out_76.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/F8A90D54-E79D-4E4A-9E4F-89828D987DA9.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018C-UL2018_MiniAODv2-v2/out_75.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/8272ACFA-A088-7745-AD49-759EA6111EDA.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018C-UL2018_MiniAODv2-v2/out_74.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/48866064-A8CA-5146-A380-D806769C5201.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018C-UL2018_MiniAODv2-v2/out_73.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/D13C3C99-8E3F-C647-8C8B-41E5027B899A.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018C-UL2018_MiniAODv2-v2/out_72.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/14A091A9-998F-874C-AA00-A12DE31BCC00.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018C-UL2018_MiniAODv2-v2/out_71.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/74DB32C8-E119-1C42-9A67-DC315BD1B402.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018C-UL2018_MiniAODv2-v2/out_70.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/E2653DA7-0F07-0C4C-BD83-B2E378064E40.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018C-UL2018_MiniAODv2-v2/out_655.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/AB44EF6D-4859-2740-832D-ADB3072349EA.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018C-UL2018_MiniAODv2-v2/out_631.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/3925F337-ABD1-7746-9D88-431FBA550AEE.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018C-UL2018_MiniAODv2-v2/out_567.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/30271F58-A224-FE40-A6DA-C0D6E43CA1F7.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018C-UL2018_MiniAODv2-v2/out_79.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/EC9CDA62-5BB9-914F-BBA2-998A2C0655BA.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018C-UL2018_MiniAODv2-v2/out_78.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70001/4456088F-A121-454F-8033-856F99DAD68F.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018C-UL2018_MiniAODv2-v2/out_1043.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/5D1366FA-90BA-E740-A2CF-641D61E1674A.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018C-UL2018_MiniAODv2-v2/out_1042.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/A1D9D6F5-B984-3E45-A563-75EA8A505B7A.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018C-UL2018_MiniAODv2-v2/out_1041.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/837F7321-B0ED-6E41-97D0-3A2AA0523B72.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018C-UL2018_MiniAODv2-v2/out_1040.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/B1C67C50-0A6B-AF44-839B-52852074F9BA.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018C-UL2018_MiniAODv2-v2/out_1047.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/BC4223B5-2C18-F147-8288-836A6036ACFB.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018C-UL2018_MiniAODv2-v2/out_1046.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70001/9F082A1F-237E-BC43-B90C-7FA8085700F3.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018C-UL2018_MiniAODv2-v2/out_1045.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70001/3F11AE83-3A06-754E-AE47-530A6D982CE6.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018C-UL2018_MiniAODv2-v2/out_1044.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70001/F5D7D743-A93D-3448-9FA3-5562582124D8.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018C-UL2018_MiniAODv2-v2/out_1049.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70001/499E2877-7776-D640-83FF-B251B3ABF7EE.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018C-UL2018_MiniAODv2-v2/out_1048.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/DDDF8858-4F23-9443-8D25-A9882ADD245A.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018C-UL2018_MiniAODv2-v2/out_1222.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70002/85260F60-9832-4545-B23D-7713CFA64064.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018C-UL2018_MiniAODv2-v2/out_359.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70002/508CFEF0-CE58-8541-9A8E-C718204A41BB.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018C-UL2018_MiniAODv2-v2/out_358.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/83F1AA82-ED29-F541-819A-3F8323A69C65.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018C-UL2018_MiniAODv2-v2/out_868.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70001/15308A86-4444-BC44-8733-66B8D8781A28.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018C-UL2018_MiniAODv2-v2/out_1142.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70001/22180F2D-C2C4-F64E-8D74-B8B02C25216E.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018C-UL2018_MiniAODv2-v2/out_1143.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/351CF77F-52CC-1948-912C-D798410A09CC.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018C-UL2018_MiniAODv2-v2/out_1140.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/3E48DA5E-93AF-F34D-8F57-7342ACC556B6.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018C-UL2018_MiniAODv2-v2/out_1141.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70001/B74A0A54-9801-7540-9E7A-E0A3ACC6CD53.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018C-UL2018_MiniAODv2-v2/out_1146.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70001/3B8EB45C-7FA6-CA4E-B5CE-5A8F2110FDA7.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018C-UL2018_MiniAODv2-v2/out_1147.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70001/B2C2EBF1-940A-B74A-93D1-011CB710273B.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018C-UL2018_MiniAODv2-v2/out_1144.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70001/1744BCE9-AF81-2D4B-9C4B-37A08FF5F4A2.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018C-UL2018_MiniAODv2-v2/out_1145.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/7CEC3A36-3EE3-8742-817E-4BEAB8BF1F87.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018C-UL2018_MiniAODv2-v2/out_667.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/7082C189-E546-2549-8774-B57EEF8EB00F.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018C-UL2018_MiniAODv2-v2/out_666.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70001/EB2714AB-15EC-7346-9BAA-E3D5DD3F4E82.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018C-UL2018_MiniAODv2-v2/out_1148.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70001/FC5F1A53-3D20-994A-B47C-F643C8300A63.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018C-UL2018_MiniAODv2-v2/out_1149.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/C8A44B3C-391A-7D42-8297-D6C3601AFD95.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018C-UL2018_MiniAODv2-v2/out_663.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/FCF64DB4-3BA2-5548-BBB6-B237227AC0CF.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018C-UL2018_MiniAODv2-v2/out_662.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/EC463C1E-36E9-624E-BDF2-E371FC3E83BB.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018C-UL2018_MiniAODv2-v2/out_661.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/8657729B-97BF-7D4F-8777-D9D792408854.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018C-UL2018_MiniAODv2-v2/out_660.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/41AC4EC0-F501-B243-A594-A67B6BF70D81.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018C-UL2018_MiniAODv2-v2/out_1221.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/6D7C0E30-FCE3-AD47-A1EC-F22CF4C15DD0.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018C-UL2018_MiniAODv2-v2/out_215.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/BF360017-B2C1-E745-8AD1-41EBC40CA572.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018C-UL2018_MiniAODv2-v2/out_653.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/866C153F-B0CD-DD48-BCFF-777F972F59E7.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018C-UL2018_MiniAODv2-v2/out_1037.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/9D68E439-0812-AA47-B4A5-A34F4BB282D5.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018C-UL2018_MiniAODv2-v2/out_692.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/457664EC-8644-1044-A18F-0C2F87926C03.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018C-UL2018_MiniAODv2-v2/out_693.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/9A69263B-5692-2B4E-B5C6-5EAD61DCAA30.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018C-UL2018_MiniAODv2-v2/out_690.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/BDC4D916-127D-344B-9B19-835F06CC4602.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018C-UL2018_MiniAODv2-v2/out_691.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/FDD72784-59B3-4C42-B7EA-BA0A76D6E67C.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018C-UL2018_MiniAODv2-v2/out_696.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/08A89976-7F56-2D45-9A20-CD4356500B90.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018C-UL2018_MiniAODv2-v2/out_697.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/992D9646-B35B-1048-A635-324F8834FCBA.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018C-UL2018_MiniAODv2-v2/out_694.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/65363BCF-BAED-054A-8D9E-19603688A03D.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018C-UL2018_MiniAODv2-v2/out_695.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/5CA27F4C-80DD-3148-BC24-B6D012F00614.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018C-UL2018_MiniAODv2-v2/out_698.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/2B4FFFAE-2047-694B-8344-226D5704C29D.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018C-UL2018_MiniAODv2-v2/out_699.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/71C7C48E-EB89-F24D-8F3C-B91ADDA6E8F4.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018C-UL2018_MiniAODv2-v2/out_222.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70001/F0453F67-AD90-F144-B73A-A406A8CAD87A.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018C-UL2018_MiniAODv2-v2/out_542.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70001/9958B49E-A208-3545-AF63-212202F7368C.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018C-UL2018_MiniAODv2-v2/out_543.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70001/3411D027-1989-344E-929A-9835B9996015.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018C-UL2018_MiniAODv2-v2/out_540.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70001/7B273913-950F-E948-9C76-9CE20E8784C5.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018C-UL2018_MiniAODv2-v2/out_541.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/10D78C43-650D-0841-B8C5-82B66C1E20BB.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018C-UL2018_MiniAODv2-v2/out_546.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/3D72DD8B-F0AE-FD41-B558-15CD89EF712D.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018C-UL2018_MiniAODv2-v2/out_547.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/F0DE2101-D393-6342-9077-462D31C5D1A0.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018C-UL2018_MiniAODv2-v2/out_544.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/1C589372-8333-B045-9000-A2C2D822E0F5.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018C-UL2018_MiniAODv2-v2/out_545.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/88303AC3-0204-3F42-9BD6-7B0413307BC1.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018C-UL2018_MiniAODv2-v2/out_1018.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/CE3B5DAA-16C6-4845-979C-DE1796F398A0.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018C-UL2018_MiniAODv2-v2/out_548.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/3090EB11-2128-F849-A01A-C6305E6BEFA7.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018C-UL2018_MiniAODv2-v2/out_549.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/25BCF357-EFFB-3E48-99C9-A7F7188903DF.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018C-UL2018_MiniAODv2-v2/out_827.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/E4E6A669-5EAF-574C-97C1-987117A42598.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018C-UL2018_MiniAODv2-v2/out_1019.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70001/7CA41D7D-1E79-CB48-927A-23687CF04930.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018C-UL2018_MiniAODv2-v2/out_995.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70001/05A9F768-0444-2C43-8852-57E33C85EED7.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018C-UL2018_MiniAODv2-v2/out_994.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70001/7BFB0DAE-B19E-8348-9B61-3F14F8390C62.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018C-UL2018_MiniAODv2-v2/out_997.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70001/FB6DD8C0-3681-8348-AC75-B6BF8A579B4F.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018C-UL2018_MiniAODv2-v2/out_996.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70001/33802D66-C720-9048-9595-A4A92B04E1FF.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018C-UL2018_MiniAODv2-v2/out_991.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70001/59CFFDBB-D139-8B47-AD98-116406CACBC1.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018C-UL2018_MiniAODv2-v2/out_990.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70001/764972E0-E1F2-0C45-B60C-1F63AAAE35B7.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018C-UL2018_MiniAODv2-v2/out_993.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70001/DD29B6CB-FCB3-E849-8352-1C2B447680B4.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018C-UL2018_MiniAODv2-v2/out_992.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/B0ACDB42-B7D6-2E41-8E16-F89B46BC31EC.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018C-UL2018_MiniAODv2-v2/out_999.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/9602D813-1A7A-A442-B182-FF1B94E25C42.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018C-UL2018_MiniAODv2-v2/out_998.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70001/3291C0F4-D8FC-6E40-8E5D-C33510FFFFA7.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018C-UL2018_MiniAODv2-v2/out_120.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70001/EFD4FC78-80B6-4C4E-A819-72FB37590A43.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018C-UL2018_MiniAODv2-v2/out_121.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70001/AF2ABD11-6E28-8A4F-81F6-9ABFA43661AC.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018C-UL2018_MiniAODv2-v2/out_122.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/DE44D8BC-F554-4D44-879D-167AB8FBA99D.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018C-UL2018_MiniAODv2-v2/out_123.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/3CD5EB87-C88D-0642-8F1E-E56D90C2C8AA.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018C-UL2018_MiniAODv2-v2/out_124.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70001/3D239357-A5F9-614C-9185-FC9F2253B3E4.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018C-UL2018_MiniAODv2-v2/out_125.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70001/C365C17F-31BC-4743-AD2C-D6667440BBEB.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018C-UL2018_MiniAODv2-v2/out_126.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70001/B6E1992E-9C46-FA47-8250-F3F989EC4538.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018C-UL2018_MiniAODv2-v2/out_127.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70001/F3B0448D-84F8-F94C-9813-C1913A1296B7.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018C-UL2018_MiniAODv2-v2/out_128.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/0CA8FF23-88D9-F745-B816-B9C2D6203196.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018C-UL2018_MiniAODv2-v2/out_2.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/9CDCB972-BD65-3644-A11E-6D12908B238F.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018C-UL2018_MiniAODv2-v2/out_219.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/74A27161-AF7B-D447-8AAB-DE1861CE3D20.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018C-UL2018_MiniAODv2-v2/out_176.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/C95BEE6E-812F-3242-AE99-60C4E14F2270.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018C-UL2018_MiniAODv2-v2/out_214.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70001/33D984A6-F85B-EE42-94E1-2E3A1D0F1716.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018C-UL2018_MiniAODv2-v2/out_777.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/E7FAC2A9-F6A0-B74D-9718-86ACBCD751FA.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018C-UL2018_MiniAODv2-v2/out_1128.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/DAFB7603-E9DF-E64E-8773-BC679FA6F9D4.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018C-UL2018_MiniAODv2-v2/out_1010.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/6068D7F9-4D49-7A40-AC28-21BBA3D44BEF.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018C-UL2018_MiniAODv2-v2/out_563.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/A582B873-7AED-204D-8541-C862415CC80F.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018C-UL2018_MiniAODv2-v2/out_1022.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/41BD9920-7E14-5F42-BEBC-BF3AC5F0BA76.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018C-UL2018_MiniAODv2-v2/out_1011.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/0FF934FD-07DB-234C-BC0C-D0FF64AAA91E.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018C-UL2018_MiniAODv2-v2/out_928.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70001/8FA37B61-835A-4247-BBE3-9E16FA4E99E3.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018C-UL2018_MiniAODv2-v2/out_929.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/A7178983-A9AF-8648-9280-41FF3BAA56CD.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018C-UL2018_MiniAODv2-v2/out_416.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/C1B86BAE-88E5-AD4C-B482-6CD6A9BBF4A6.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018C-UL2018_MiniAODv2-v2/out_417.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/81C7931D-1307-1D4E-840F-0A5D93F5677D.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018C-UL2018_MiniAODv2-v2/out_410.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/8F8E0AD5-2E7B-DA42-9FB0-4D447536604A.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018C-UL2018_MiniAODv2-v2/out_411.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/8DA8D340-A592-A347-954C-F1AA96A3620C.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018C-UL2018_MiniAODv2-v2/out_412.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/42B1BB2F-F933-8241-96F0-902E49BADD13.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018C-UL2018_MiniAODv2-v2/out_413.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/5E4C67AD-CCDF-4747-BE38-354F2FF3EA3E.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018C-UL2018_MiniAODv2-v2/out_920.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/1B285B86-4578-E34D-BB92-74EAFF1D0C9E.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018C-UL2018_MiniAODv2-v2/out_498.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/867F8972-6B15-EA4A-9641-B11114325B68.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018C-UL2018_MiniAODv2-v2/out_922.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/33FD738D-C927-774A-B456-9DBD44C55FF6.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018C-UL2018_MiniAODv2-v2/out_923.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/72EE3FC6-2A42-9941-A169-62FABC65865C.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018C-UL2018_MiniAODv2-v2/out_924.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/E9433CA4-962B-DA42-BFF5-A1FEB0D8F23F.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018C-UL2018_MiniAODv2-v2/out_925.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/E09552C2-19F0-FF4E-A878-2FC329485E09.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018C-UL2018_MiniAODv2-v2/out_926.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/0E8359F8-5566-E745-A2FB-E3F4F5C0EA0F.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018C-UL2018_MiniAODv2-v2/out_927.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70002/D78CABB0-F275-6244-AD99-7E6FEEECED33.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018C-UL2018_MiniAODv2-v2/out_319.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70002/03E29583-4148-6842-A1D6-3088F0872195.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018C-UL2018_MiniAODv2-v2/out_318.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/69E91B15-6C9C-A449-872F-28F631C395F4.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018C-UL2018_MiniAODv2-v2/out_313.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/CF4E4C2D-C6F1-9140-9E7C-CC9139455C22.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018C-UL2018_MiniAODv2-v2/out_312.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/19231F65-9AEB-0C42-8BEB-610A91DAA442.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018C-UL2018_MiniAODv2-v2/out_311.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/2B8BD57F-598A-D744-8549-F23FC0CDB1F4.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018C-UL2018_MiniAODv2-v2/out_310.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70002/CD45E149-7242-5244-8B46-37C96D82EEB5.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018C-UL2018_MiniAODv2-v2/out_317.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/A3C10B0D-4AD1-AC45-AF59-5288B19D9F67.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018C-UL2018_MiniAODv2-v2/out_449.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/8A25961F-65D9-8749-9BD1-433C0E5D1EA9.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018C-UL2018_MiniAODv2-v2/out_315.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/2E7121FF-C828-1242-96D4-2C0B9097EA88.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018C-UL2018_MiniAODv2-v2/out_1028.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/BC0DC019-2408-3842-80FF-D86EB0869108.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018C-UL2018_MiniAODv2-v2/out_1237.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/1AA12C0B-7758-E946-A915-3557C1AEAF5A.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018C-UL2018_MiniAODv2-v2/out_921.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/4D56BDC0-1A1C-504B-A6EB-FC952D522C59.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018C-UL2018_MiniAODv2-v2/out_496.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/E09BDBA8-0DA7-1044-BED8-0ECF43C0D7AC.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018C-UL2018_MiniAODv2-v2/out_1134.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/D02B1410-ACCC-A34E-A580-6F7C9EB26D99.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018C-UL2018_MiniAODv2-v2/out_832.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/A010E111-7095-8440-9A3A-4B4BE1F14DA0.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018C-UL2018_MiniAODv2-v2/out_833.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/B9A578EC-0270-1F49-9B78-66C5DE6F2008.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018C-UL2018_MiniAODv2-v2/out_830.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/1951EBED-E642-404C-BE74-CD3BBA244A06.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018C-UL2018_MiniAODv2-v2/out_831.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/25814F02-018D-3C4A-B0F7-B02156F3B7EE.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018C-UL2018_MiniAODv2-v2/out_836.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/43213E7A-3870-C844-8FC6-1FAA35216DDA.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018C-UL2018_MiniAODv2-v2/out_837.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/C15701D9-8207-B74E-BCC1-8618E4F00DAB.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018C-UL2018_MiniAODv2-v2/out_834.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/59816636-7F12-1D47-9BB4-60FC361EA456.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018C-UL2018_MiniAODv2-v2/out_835.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/661FA1DC-5A51-8B47-8D72-CEAA3F344D75.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018C-UL2018_MiniAODv2-v2/out_838.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/0FED116F-EB2D-174A-9144-07A7BAA70F74.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018C-UL2018_MiniAODv2-v2/out_839.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/8E69B8E0-206E-9043-9AD0-714ABAA86D2D.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018C-UL2018_MiniAODv2-v2/out_3.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/EBD94583-791B-DD41-ADF0-F7D53D8C6EE1.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018C-UL2018_MiniAODv2-v2/out_725.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70002/119E3CC3-F3F2-B047-91A2-EAC014FA0E0E.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018C-UL2018_MiniAODv2-v2/out_368.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/4C95D558-BD61-9D41-B320-9DF13C63DB2B.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018C-UL2018_MiniAODv2-v2/out_369.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70002/2F49CD42-CF5D-4C4D-949C-0B95E439693F.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018C-UL2018_MiniAODv2-v2/out_366.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70002/E225E27A-CEAA-4E48-B4C5-EAC3815C4F5D.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018C-UL2018_MiniAODv2-v2/out_367.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70002/D4CAA6D9-EE3E-8343-96AE-3F1356C0BD8B.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018C-UL2018_MiniAODv2-v2/out_364.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70002/22A03066-8D84-D94D-A41B-6DBFFF49E1C2.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018C-UL2018_MiniAODv2-v2/out_365.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/53C6442C-66D5-7649-A700-300425971218.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018C-UL2018_MiniAODv2-v2/out_362.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70002/41A67F12-7026-D947-AFED-EC112613EB59.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018C-UL2018_MiniAODv2-v2/out_363.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/9B4E21EE-A818-D649-81D2-1104F9D1645E.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018C-UL2018_MiniAODv2-v2/out_360.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/B6348797-8FCA-454E-A5F6-33C195CFA09D.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018C-UL2018_MiniAODv2-v2/out_361.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/009485C6-7252-6E48-A482-6E25778A055D.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018C-UL2018_MiniAODv2-v2/out_720.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/7B7AD37D-9B94-C249-94AF-FB71A51A216C.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018C-UL2018_MiniAODv2-v2/out_218.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/2766A93A-00AA-E747-8D1D-22528D5B06B9.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018C-UL2018_MiniAODv2-v2/out_1238.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/9FF7E97F-89AE-724C-9031-79E947C6E06B.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018C-UL2018_MiniAODv2-v2/out_1239.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/A3C478FF-5E42-DA48-A2E4-CADA2BFA2797.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018C-UL2018_MiniAODv2-v2/out_380.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/DB85D008-AD8B-FF40-A353-4C0E68A0A3DB.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018C-UL2018_MiniAODv2-v2/out_861.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/CADD14F4-4C9E-B04B-89CC-5880EEB7F7DE.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018C-UL2018_MiniAODv2-v2/out_382.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/AFFABC7D-5F73-3A4D-9E19-C3BABC00B817.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018C-UL2018_MiniAODv2-v2/out_383.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/C1EEBEC1-1550-FB49-B1F9-1C16391605CA.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018C-UL2018_MiniAODv2-v2/out_384.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/3B7786E5-9780-9549-AB0C-4894698590EF.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018C-UL2018_MiniAODv2-v2/out_385.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/203051B5-E530-2740-8188-94624736FB5F.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018C-UL2018_MiniAODv2-v2/out_386.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/72189E45-9D16-3C4B-9FE3-68F55D7A7337.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018C-UL2018_MiniAODv2-v2/out_387.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/072AD293-70CF-CB45-9AC5-C753BB1E5D11.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018C-UL2018_MiniAODv2-v2/out_388.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/2600BD91-7D1E-6345-9721-47BBA2BD649F.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018C-UL2018_MiniAODv2-v2/out_389.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/85E225B0-A762-0F4A-B467-D8F7C0762539.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018C-UL2018_MiniAODv2-v2/out_784.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70001/A142CB3E-F8ED-8B4E-819A-CB5AB8A767ED.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018C-UL2018_MiniAODv2-v2/out_785.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70001/C7CDAD7A-31F7-914B-95D4-9DD3BC285A36.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018C-UL2018_MiniAODv2-v2/out_786.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70001/D6DE6203-B1C2-1E4D-A4AE-606485744A50.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018C-UL2018_MiniAODv2-v2/out_787.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/3446EFC3-8CDC-F24E-A3AE-F529D1B991C3.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018C-UL2018_MiniAODv2-v2/out_780.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/3216651B-40B1-934F-91BB-201F0BBFF252.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018C-UL2018_MiniAODv2-v2/out_781.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/B84E5AED-E761-914D-918B-BE88BE073C35.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018C-UL2018_MiniAODv2-v2/out_782.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/6A68A2FE-1CCF-FF4A-ABFE-2CBF2B5EF86E.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018C-UL2018_MiniAODv2-v2/out_381.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70001/01FB6340-1859-AE45-A82B-C0E389F84D98.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018C-UL2018_MiniAODv2-v2/out_788.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/BA18E673-779E-F243-9836-B6FCB0BEDF12.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018C-UL2018_MiniAODv2-v2/out_789.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70001/F61FE815-37F5-544A-91E2-17CFC95B2D18.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018C-UL2018_MiniAODv2-v2/out_1174.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/FD9E5A86-7ABE-F249-8653-9C3FAA0DF9D7.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018C-UL2018_MiniAODv2-v2/out_860.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/B1C8E674-46D3-314B-BCD5-58A3FE594ACD.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018C-UL2018_MiniAODv2-v2/out_630.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/DD4331D1-EE74-5B4C-933E-FBDC64BA6ED6.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018C-UL2018_MiniAODv2-v2/out_114.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/E80F19A1-6ACF-5A41-91D9-F1EBFD20F2EE.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018C-UL2018_MiniAODv2-v2/out_1223.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/A9B71FF7-0CA6-8F48-A615-D9CDA9C85DE3.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018C-UL2018_MiniAODv2-v2/out_1164.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70001/5ADCDCEA-AEE7-834A-962D-D37CB9BCD1CF.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018C-UL2018_MiniAODv2-v2/out_579.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70001/463C9DA9-B5BA-2E43-87F3-4FADA6EDD9C4.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018C-UL2018_MiniAODv2-v2/out_578.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/7C0DA16E-9463-374A-BFB3-D97B0A868802.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018C-UL2018_MiniAODv2-v2/out_1165.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/AAF189FF-32AF-3042-ACAB-AC3BE8A1D821.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018C-UL2018_MiniAODv2-v2/out_573.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70001/10F8717D-A430-F748-8F81-00C452908BAD.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018C-UL2018_MiniAODv2-v2/out_572.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70001/F1AA2B7C-007B-654A-B533-872AC73507FD.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018C-UL2018_MiniAODv2-v2/out_571.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/12F9D9BE-F441-B144-8A53-755D566269DB.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018C-UL2018_MiniAODv2-v2/out_570.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70001/A2C9B0B6-EFE3-3049-88A5-5A8485F36CB0.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018C-UL2018_MiniAODv2-v2/out_577.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/137B7C56-61D7-3F4D-9C1D-AABFD7774B9E.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018C-UL2018_MiniAODv2-v2/out_576.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/C15D9A7E-A9AC-7A4C-9442-67BCB93B3A33.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018C-UL2018_MiniAODv2-v2/out_575.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/20CE1689-1828-D446-88C6-1C8B27656C7A.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018C-UL2018_MiniAODv2-v2/out_574.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/BE2E82E9-C994-5A4C-B064-01854B0BFE37.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018C-UL2018_MiniAODv2-v2/out_60.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/73DD7F25-D444-EE48-9AED-825E04A4179C.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018C-UL2018_MiniAODv2-v2/out_61.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/B92E0619-BC5A-B54D-9FE7-9A76C0BE408B.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018C-UL2018_MiniAODv2-v2/out_62.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70002/EDCC1B90-0C96-554F-8EB6-BC26282D72CA.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018C-UL2018_MiniAODv2-v2/out_259.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/F932AFCA-B5C4-1243-ADD0-547474ADE68A.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018C-UL2018_MiniAODv2-v2/out_64.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/20C1717E-F727-584B-8649-9DB13DBED1E2.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018C-UL2018_MiniAODv2-v2/out_65.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/A5B71846-EF06-BC4D-97E5-53C4B3271386.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018C-UL2018_MiniAODv2-v2/out_66.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/E0E1BD50-9494-C047-9DD2-0954253114CB.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018C-UL2018_MiniAODv2-v2/out_67.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/0C02C89B-7C96-AB40-8C6A-AB2CDFED47C8.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018C-UL2018_MiniAODv2-v2/out_68.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/9D4C6D38-8C31-DC41-AD73-76CC71066316.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018C-UL2018_MiniAODv2-v2/out_253.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70002/DB2F4A2A-D0C2-F148-8F66-761C72038709.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018C-UL2018_MiniAODv2-v2/out_250.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70002/8FC5425F-40FF-7441-9E86-CA074F25089F.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018C-UL2018_MiniAODv2-v2/out_251.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70002/D7231D96-6247-4B4C-9141-F8933B43F0DC.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018C-UL2018_MiniAODv2-v2/out_256.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70002/A33E7140-BF32-8D47-B473-2734E5AD5A0E.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018C-UL2018_MiniAODv2-v2/out_257.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70002/3A1B90E4-76AB-2348-BC4B-7BE99389F019.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018C-UL2018_MiniAODv2-v2/out_254.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70002/96C4F025-F1D2-1F4B-96DE-7D327BBE3483.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018C-UL2018_MiniAODv2-v2/out_255.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/734E9EEB-86D9-F846-A842-93E53AD8DCD9.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018C-UL2018_MiniAODv2-v2/out_499.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/162BA203-E73D-F74F-97DA-C918272C58B7.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018C-UL2018_MiniAODv2-v2/out_603.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70001/FC9F2461-BE88-F34F-9D08-00B9E31C12B3.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018C-UL2018_MiniAODv2-v2/out_1172.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/168F5D75-E8D9-BE4E-8C30-80E3A47C48F6.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018C-UL2018_MiniAODv2-v2/out_156.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/BD078AE7-13CA-DA46-BBCD-B811698FA3DA.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018C-UL2018_MiniAODv2-v2/out_939.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70001/79FDBF42-D72D-F745-8888-8DF10716FFCB.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018C-UL2018_MiniAODv2-v2/out_731.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/C5AFB1CB-2244-9247-8E8A-76BA6BE3A07E.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018C-UL2018_MiniAODv2-v2/out_730.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70001/EFF32A66-B132-BA4C-BE91-569828960113.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018C-UL2018_MiniAODv2-v2/out_733.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/883806FA-6643-774B-BEE3-C8DEA77EC50B.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018C-UL2018_MiniAODv2-v2/out_938.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70001/B3DBB2F8-12B3-6F47-A1AF-70DDB6AC01D9.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018C-UL2018_MiniAODv2-v2/out_735.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70001/94A1D2F0-2B5B-A943-86B4-C410DFF14F7D.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018C-UL2018_MiniAODv2-v2/out_734.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70001/E8703C7E-72EA-004E-A8D3-C8C9F6849BCC.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018C-UL2018_MiniAODv2-v2/out_508.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70001/16BF6F81-92E5-9147-B1B7-E97A222DED03.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018C-UL2018_MiniAODv2-v2/out_736.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/849775EE-EB86-3641-8742-014037D54DAE.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018C-UL2018_MiniAODv2-v2/out_1032.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/FD0D79C1-1E24-2C43-AFD7-DD2BE8D14632.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018C-UL2018_MiniAODv2-v2/out_738.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/3642AE72-540A-3A46-87BD-2420A69B625D.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018C-UL2018_MiniAODv2-v2/out_1030.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70001/22184C04-B878-FA4D-8695-E3D28AC4AE23.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018C-UL2018_MiniAODv2-v2/out_505.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/6FE20D25-4E48-9C43-8BB2-F1CC6F9B24B7.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018C-UL2018_MiniAODv2-v2/out_502.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/4BF51A12-13DF-C14D-8E06-2CA23805BF48.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018C-UL2018_MiniAODv2-v2/out_503.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/91870F79-FCD1-414A-AD82-5D57B0FC94C0.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018C-UL2018_MiniAODv2-v2/out_500.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/1F70FB25-FE4C-674A-8E67-35574D08CCEF.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018C-UL2018_MiniAODv2-v2/out_501.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/67BB72C6-CB42-8542-8690-1C1F5E30F1B9.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018C-UL2018_MiniAODv2-v2/out_1212.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/A634BAD7-3CFB-3E4E-A2A4-A7E30FA71286.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018C-UL2018_MiniAODv2-v2/out_1213.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/2FAE7398-0B08-0943-964A-3D1FA879D0C4.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018C-UL2018_MiniAODv2-v2/out_632.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/2476CE17-6D80-C146-99DB-5856CF988C0E.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018C-UL2018_MiniAODv2-v2/out_633.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/75537638-6DCA-4143-9516-F2A0F1642027.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018C-UL2018_MiniAODv2-v2/out_1216.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70001/8098FFAF-99B9-BD4B-BFFA-5E2378850655.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018C-UL2018_MiniAODv2-v2/out_468.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/CB952D0B-2A15-AA4B-B3AC-DB88FAA7970C.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018C-UL2018_MiniAODv2-v2/out_636.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/81D6AF52-B525-E54A-910E-0C294DD88AF7.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018C-UL2018_MiniAODv2-v2/out_637.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/6A66C8CA-AEED-124F-8C0E-3B7259173378.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018C-UL2018_MiniAODv2-v2/out_465.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/1E0D697A-175C-0247-9243-A961897AFC95.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018C-UL2018_MiniAODv2-v2/out_464.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/75D95E4A-0795-D34E-9827-1BFBB8C44E29.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018C-UL2018_MiniAODv2-v2/out_1218.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/E2552F54-AACF-9B4E-8BDE-D6173A456A36.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018C-UL2018_MiniAODv2-v2/out_466.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/37F0A6ED-ED64-AE40-81D8-8A4ED0B98AFF.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018C-UL2018_MiniAODv2-v2/out_461.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/80CB12CB-6BB6-CA49-8895-4313663E8156.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018C-UL2018_MiniAODv2-v2/out_900.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/7D79E97A-8C56-544D-A602-82C0FF0451AE.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018C-UL2018_MiniAODv2-v2/out_463.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/4573CD93-B78E-6E40-92F5-50AEFF721634.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018C-UL2018_MiniAODv2-v2/out_462.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70002/B527BC45-82B6-0847-A8E9-7B873C52A1B8.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018C-UL2018_MiniAODv2-v2/out_281.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/85F3A183-25A0-C64F-8ECD-79AD33DB4690.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018C-UL2018_MiniAODv2-v2/out_901.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/3FEC90A8-24F1-7546-885F-95DF802E3C4C.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018C-UL2018_MiniAODv2-v2/out_168.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/20B6F54B-D677-5F4F-805C-33D8E1E7B40D.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018C-UL2018_MiniAODv2-v2/out_169.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/D81268E1-120D-E843-BC2F-98B1FA18C0B6.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018C-UL2018_MiniAODv2-v2/out_164.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/41B1ACD8-3D25-2742-90E8-3E1301AE2E5D.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018C-UL2018_MiniAODv2-v2/out_165.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/D8C99E13-2471-794D-8455-F4DEAEF7C09C.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018C-UL2018_MiniAODv2-v2/out_166.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/3D609E71-A35E-2F42-AD0A-70127B6A0CB2.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018C-UL2018_MiniAODv2-v2/out_167.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/96090148-294E-DE48-B1DF-8205A2E48693.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018C-UL2018_MiniAODv2-v2/out_160.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/F3B32EA8-7FA1-284B-9A2A-3E9CE3DF189D.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018C-UL2018_MiniAODv2-v2/out_161.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/ED22E5B7-150C-3F44-918D-CED2D7AF8C3F.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018C-UL2018_MiniAODv2-v2/out_1016.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/A9B91E09-CB9C-A948-9B86-9538098867BE.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018C-UL2018_MiniAODv2-v2/out_163.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70001/77E65C68-8618-E04A-9769-00DE5038DBCC.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018C-UL2018_MiniAODv2-v2/out_964.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70001/7748452E-C0E9-4640-9C80-E0FF392AA17C.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018C-UL2018_MiniAODv2-v2/out_965.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70001/496FC198-3478-E548-B2C7-6F9F3B32EE2F.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018C-UL2018_MiniAODv2-v2/out_966.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70001/E0F77925-4DAF-294C-96D5-7788022B3296.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018C-UL2018_MiniAODv2-v2/out_967.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/C311B99F-A347-1C4A-9FDB-3661EF45D805.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018C-UL2018_MiniAODv2-v2/out_960.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/BF97A3DC-6947-4342-B728-4DF3B787DD11.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018C-UL2018_MiniAODv2-v2/out_961.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/33791BC7-0422-4C43-AA52-889949B63DE1.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018C-UL2018_MiniAODv2-v2/out_962.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/39A6E783-6AF1-D940-BCED-670EB97EA3E7.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018C-UL2018_MiniAODv2-v2/out_963.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/B99D8178-C2B8-D540-B156-C27A15488C00.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018C-UL2018_MiniAODv2-v2/out_1129.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/61C3693C-BCB2-E64C-83B3-C1C03FC6B219.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018C-UL2018_MiniAODv2-v2/out_1039.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/99E592F3-53DC-BE4C-9F1D-0646F511B5F1.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018C-UL2018_MiniAODv2-v2/out_783.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70001/95BE5E2B-D2A8-5D42-BD4C-EA2512FA6848.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018C-UL2018_MiniAODv2-v2/out_968.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70001/511BB95B-8644-F349-8D93-1D7810A47482.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018C-UL2018_MiniAODv2-v2/out_969.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/0373DDC1-4855-4144-B75B-453D77AAC3D0.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018C-UL2018_MiniAODv2-v2/out_936.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/A8DDF3E0-A10A-B041-95FB-FFBA890E8BFA.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018C-UL2018_MiniAODv2-v2/out_1106.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/65FEEDD7-BF1A-344C-A0CC-00ADE1E5A793.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018C-UL2018_MiniAODv2-v2/out_1107.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/A070E989-13EA-384B-ABB3-89E49BF58084.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018C-UL2018_MiniAODv2-v2/out_1104.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/B9078175-C994-B94E-9A0F-9350970CA635.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018C-UL2018_MiniAODv2-v2/out_1105.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/0C8CAE67-7779-D842-A4D4-4BBCCDB3E06A.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018C-UL2018_MiniAODv2-v2/out_1102.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/DF6EA7FC-5848-0D44-B2F0-55BAFF5D3B63.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018C-UL2018_MiniAODv2-v2/out_1103.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/B50CA72C-22FA-154F-A52F-1C080723DDA8.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018C-UL2018_MiniAODv2-v2/out_1100.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/036239FA-1662-4F48-9026-2FC0CD3B08EB.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018C-UL2018_MiniAODv2-v2/out_1101.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/E51FE8F4-AFDA-8E45-BCF1-C24F46550A03.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018C-UL2018_MiniAODv2-v2/out_934.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/90CC8D1C-C553-394C-8632-46D2CD9271AA.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018C-UL2018_MiniAODv2-v2/out_908.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/D2D3E861-EE57-904D-943D-21F1CB126148.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018C-UL2018_MiniAODv2-v2/out_1108.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/EA4E5896-8221-994A-86FD-935DA2845E16.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018C-UL2018_MiniAODv2-v2/out_1109.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/61FAB949-40CE-B14D-82A1-1AA0955735CB.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018C-UL2018_MiniAODv2-v2/out_670.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/1D4E9171-8389-F04B-A2EC-EC519DF0BDE0.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018C-UL2018_MiniAODv2-v2/out_909.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/A6C9B225-4582-0E4D-976D-2F642860BC38.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018C-UL2018_MiniAODv2-v2/out_635.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/91CEE6E3-221D-304D-AC94-BF8A1BE94B42.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018C-UL2018_MiniAODv2-v2/out_722.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/16C7715A-EC2D-FA4A-B8A4-B4EF76649F7C.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018C-UL2018_MiniAODv2-v2/out_584.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/6F73214C-AD71-C748-804E-719837B56EB4.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018C-UL2018_MiniAODv2-v2/out_678.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/331E47B4-3155-F945-8B78-8525329B82B0.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018C-UL2018_MiniAODv2-v2/out_609.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/5195FFAF-8827-9245-BD74-B8ECCD608D91.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018C-UL2018_MiniAODv2-v2/out_1158.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/130000/3DF3944A-36D3-FC41-A7A0-C91A05B79081.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018C-UL2018_MiniAODv2-v2/out_1241.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/6945679E-5CED-DD4C-AE2C-89387175B6D6.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018C-UL2018_MiniAODv2-v2/out_878.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/3FB8D115-186D-3248-ABBF-B25F817CB94B.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018C-UL2018_MiniAODv2-v2/out_640.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/4199FC37-9DCD-4141-A848-2DC3839B1736.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018C-UL2018_MiniAODv2-v2/out_876.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/F40C89DF-EB36-7A4C-AA67-39BC438CB3EC.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018C-UL2018_MiniAODv2-v2/out_877.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/170EF3AE-CBB8-7243-AF36-B3A61BE85A9D.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018C-UL2018_MiniAODv2-v2/out_874.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/8950938F-E459-9746-AF18-21D0F91FE259.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018C-UL2018_MiniAODv2-v2/out_875.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/48B701E5-8020-6642-B5C8-9E4CC48D07A2.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018C-UL2018_MiniAODv2-v2/out_872.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/9E355643-334E-C446-824B-CD00343F8DB5.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018C-UL2018_MiniAODv2-v2/out_643.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/79B5C25D-75A0-5D48-AEF7-B828EABE2770.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018C-UL2018_MiniAODv2-v2/out_870.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/5FB63522-3A8D-444E-89E2-6C76355CDC77.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018C-UL2018_MiniAODv2-v2/out_871.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/8C11C940-7D34-A147-8E18-3457772AB5D2.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018C-UL2018_MiniAODv2-v2/out_674.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/130000/644CB07D-0BBF-5E4B-A8CA-79FA2AA576D4.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018C-UL2018_MiniAODv2-v2/out_1242.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/403DD8C2-65EC-E442-81D3-9CC757D718D6.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018C-UL2018_MiniAODv2-v2/out_675.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/28C0B5E7-08C0-BC49-887C-C56BB8CF0CCB.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018C-UL2018_MiniAODv2-v2/out_9.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/40E3ED5D-1A25-1D4F-BFDD-6220FAC6577B.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018C-UL2018_MiniAODv2-v2/out_676.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/C090A26E-F9FD-E34D-8429-8010D5228F2F.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018C-UL2018_MiniAODv2-v2/out_890.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/FA7AE80C-8108-C141-A497-D576CA18E8EB.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018C-UL2018_MiniAODv2-v2/out_891.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/7F6C531A-D7D6-2C4B-83A7-87A9DB78AB73.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018C-UL2018_MiniAODv2-v2/out_892.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/66D879F4-03FE-8E4B-BE3C-A71F8CB501B5.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018C-UL2018_MiniAODv2-v2/out_893.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/6ED265C2-F742-624A-9469-B26CECBB5779.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018C-UL2018_MiniAODv2-v2/out_894.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/9D560093-B685-9D4C-B386-AA0EFA89FAEE.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018C-UL2018_MiniAODv2-v2/out_647.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/812C55B9-C405-0444-ABB0-11C73B898E82.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018C-UL2018_MiniAODv2-v2/out_896.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/1B674833-7109-CE4E-9FF2-D9BE055DD1B3.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018C-UL2018_MiniAODv2-v2/out_897.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/D51BACBF-70D7-1249-8B3B-DB7460082FF0.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018C-UL2018_MiniAODv2-v2/out_898.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/80CFA955-8B40-2646-BCF6-8E43D32264A2.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018C-UL2018_MiniAODv2-v2/out_899.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/E9E60202-5296-D04D-964F-4AF52510EE3C.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018C-UL2018_MiniAODv2-v2/out_1087.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/E7552012-7C33-3641-ADC9-4B2858665433.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018C-UL2018_MiniAODv2-v2/out_646.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/FF819BAE-1141-1949-B706-08E8463B513A.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018C-UL2018_MiniAODv2-v2/out_1099.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/082AF7FE-57B1-4F40-8836-05BC1DD1CC40.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018C-UL2018_MiniAODv2-v2/out_1138.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/0C825DC7-B883-C642-8388-B70C6F100584.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018C-UL2018_MiniAODv2-v2/out_1199.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/5B280080-81AF-6A40-82C7-8F16DE1FCD07.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018C-UL2018_MiniAODv2-v2/out_672.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/81ED9C28-39D0-0B44-B4CF-5B71A9107800.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018C-UL2018_MiniAODv2-v2/out_648.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/9CAD3715-61E7-984C-ABD7-8C91505172C3.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018C-UL2018_MiniAODv2-v2/out_673.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/7E84B54B-2C2F-3B45-A47A-FECEA5196875.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018C-UL2018_MiniAODv2-v2/out_1036.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/54B6BD53-FAC2-C244-B657-F5E0488CB016.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018C-UL2018_MiniAODv2-v2/out_638.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70001/A779C791-6D35-434F-9C2D-AA81E98803D1.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018C-UL2018_MiniAODv2-v2/out_1173.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/20D05595-4577-3147-8DD8-59775D03F390.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018C-UL2018_MiniAODv2-v2/out_1086.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70002/F94A8FDD-9662-BF40-973D-4BBADCA2A96E.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018C-UL2018_MiniAODv2-v2/out_357.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/6CD666B8-D9F1-8D4D-9C6B-9F0D9B3282CB.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018C-UL2018_MiniAODv2-v2/out_1198.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/39086E03-2A2D-5042-9C70-5670D2FA7346.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018C-UL2018_MiniAODv2-v2/out_809.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/0E6EF35A-622C-7B45-AAAA-9D40E836BA09.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018C-UL2018_MiniAODv2-v2/out_808.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70002/C5A2A393-68EC-C34B-9272-25A623AC3481.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018C-UL2018_MiniAODv2-v2/out_353.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70002/7509D1DD-03DB-224E-8E2E-D7325A29C75F.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018C-UL2018_MiniAODv2-v2/out_352.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70002/77E1AED3-57F1-804D-9AA5-340B5DE5D8C8.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018C-UL2018_MiniAODv2-v2/out_351.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70002/E0402C2B-3076-3D4A-9202-383B1C17614F.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018C-UL2018_MiniAODv2-v2/out_350.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/A35C4639-4D34-BA44-954C-CA664D7431E1.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018C-UL2018_MiniAODv2-v2/out_803.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/D7D94720-F24E-8347-9F0B-384FD93DC58A.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018C-UL2018_MiniAODv2-v2/out_802.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/2151A799-2C31-5B46-A11E-63DEB22027D0.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018C-UL2018_MiniAODv2-v2/out_801.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/354CC0A7-DAF0-B545-B8FB-BEF3497EB87C.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018C-UL2018_MiniAODv2-v2/out_800.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/6BF8D4C0-AEA8-2445-806F-FF0FF829D330.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018C-UL2018_MiniAODv2-v2/out_807.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/A69C50CF-CB69-BB44-90F4-91CC4D67715D.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018C-UL2018_MiniAODv2-v2/out_806.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70001/86B9D71D-8481-EB4D-84A4-DFD3BDAD2363.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018C-UL2018_MiniAODv2-v2/out_805.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70001/1A8D8B2B-E9FA-2543-AB1D-C4F85F99932D.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018C-UL2018_MiniAODv2-v2/out_804.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/B5A27AB2-6E1E-444F-8F67-5F75AF941918.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018C-UL2018_MiniAODv2-v2/out_216.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/980D3E58-741A-3344-93DA-A8B4861232A4.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018C-UL2018_MiniAODv2-v2/out_217.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70001/FD47870B-1701-1142-BDD6-3BBFF07A6BC1.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018C-UL2018_MiniAODv2-v2/out_768.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/38044469-1417-C142-8EA6-7F70270D24F1.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018C-UL2018_MiniAODv2-v2/out_769.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/686A8BF3-9521-FF4C-82ED-33C910148A7D.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018C-UL2018_MiniAODv2-v2/out_212.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/2107EB1F-4571-374E-96B0-AEE3213CE415.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018C-UL2018_MiniAODv2-v2/out_213.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/6335CCE7-5FC5-6C43-94F3-846DA7D449AA.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018C-UL2018_MiniAODv2-v2/out_210.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/0064FF45-C2E1-A242-901B-5F0248A9E1F5.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018C-UL2018_MiniAODv2-v2/out_211.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/580151AB-1702-A746-89DD-D423B49E3A40.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018C-UL2018_MiniAODv2-v2/out_762.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/F59429E3-521F-E941-83B1-E7B1D6A85F05.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018C-UL2018_MiniAODv2-v2/out_763.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/F1A3C0A8-89A7-CA40-8059-C2A6F638F5DF.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018C-UL2018_MiniAODv2-v2/out_760.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/AF6E890B-2D83-8C4A-BD48-0DCBB6FEC05E.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018C-UL2018_MiniAODv2-v2/out_761.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70001/E6040535-194E-F14E-93FF-F5137223D98E.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018C-UL2018_MiniAODv2-v2/out_766.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70001/A24A1B8F-03BE-7A42-89E4-03EC52509ECF.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018C-UL2018_MiniAODv2-v2/out_767.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/93B12F8E-CC24-2D4C-9E24-5A64E2187BAF.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018C-UL2018_MiniAODv2-v2/out_764.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70001/2FC2F339-FBCD-6C4D-80BA-11F9412AAFE8.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018C-UL2018_MiniAODv2-v2/out_765.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/45E44704-321A-774B-BF4D-75A81D3042D9.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018C-UL2018_MiniAODv2-v2/out_40.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/A6E9384F-0DAB-3D48-972C-B435E2BC5894.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018C-UL2018_MiniAODv2-v2/out_41.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/20258893-D652-BE4D-909B-CA8908194E24.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018C-UL2018_MiniAODv2-v2/out_1033.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/E0759689-908D-6343-8F24-B6748F6ACB0A.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018C-UL2018_MiniAODv2-v2/out_1169.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70001/2EFB788E-02E6-5846-9C4E-C0DAE079AAB5.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018C-UL2018_MiniAODv2-v2/out_1119.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/674F7D90-9EDB-FE4E-963B-B97163693811.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018C-UL2018_MiniAODv2-v2/out_1078.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/FB14A25D-1D6D-814A-AE77-CEDDF7AE96C5.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018C-UL2018_MiniAODv2-v2/out_1079.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/6BBC01C3-2BB4-1D45-A9D8-BFBD000A4267.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018C-UL2018_MiniAODv2-v2/out_1076.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/F186DE75-972C-024A-9517-B87157CEF9F1.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018C-UL2018_MiniAODv2-v2/out_1077.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/296252E1-D0E7-D641-8D1A-960FACFFD402.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018C-UL2018_MiniAODv2-v2/out_1074.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/10F7B296-7A12-B145-A989-B6DE890C847E.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018C-UL2018_MiniAODv2-v2/out_1075.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/EFA3DF84-D058-E24A-A877-F48B9C091894.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018C-UL2018_MiniAODv2-v2/out_1072.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/3EA2D32C-5DA9-1F46-BAF3-473EE34B87EC.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018C-UL2018_MiniAODv2-v2/out_1073.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/4A2556D0-0B6F-354B-AB14-46F57D937583.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018C-UL2018_MiniAODv2-v2/out_1070.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/AFDEA548-C986-6148-85B9-0471EBFE2B15.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018C-UL2018_MiniAODv2-v2/out_1071.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70002/5DB9FCB1-0235-E947-9C63-E701F2CD9997.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018C-UL2018_MiniAODv2-v2/out_289.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70002/45EA131A-CE6F-884D-9853-9909A18BB485.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018C-UL2018_MiniAODv2-v2/out_288.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/7D0AAA37-74C0-0B41-A0FE-C986882DD078.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018C-UL2018_MiniAODv2-v2/out_1003.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/7DE85FA3-D0C4-3947-B15F-6B665AFB5E2A.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018C-UL2018_MiniAODv2-v2/out_4.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/D335D967-5B92-8744-8895-CE7F6EFAA118.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018C-UL2018_MiniAODv2-v2/out_1179.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70002/84156D8B-745B-1640-9B4D-E58390D4E4DE.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018C-UL2018_MiniAODv2-v2/out_280.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70002/756ABD50-43E6-D74E-A85C-EF755A91B0DE.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018C-UL2018_MiniAODv2-v2/out_283.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70002/8AACC065-C952-9247-9AFB-0A51D4B17C60.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018C-UL2018_MiniAODv2-v2/out_282.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70002/69315949-A932-DF43-A5C8-339F499A6DDE.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018C-UL2018_MiniAODv2-v2/out_285.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/08E1DCCA-6F6B-5A41-89AC-D8CC22207EA7.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018C-UL2018_MiniAODv2-v2/out_284.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70002/31F34A2A-FA42-9E44-8513-94CCA8D34315.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018C-UL2018_MiniAODv2-v2/out_287.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70002/912EF7F8-84F2-2B41-B4C6-C7A0894876EB.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018C-UL2018_MiniAODv2-v2/out_286.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/FCC2BA2E-9A39-1247-9681-379E642BE8EF.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018C-UL2018_MiniAODv2-v2/out_1094.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/897907BE-2AEF-3843-BFF8-8D99B48D60DE.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018C-UL2018_MiniAODv2-v2/out_1095.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/90237EB3-E70D-B441-BAD4-CEE8E7C57DB7.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018C-UL2018_MiniAODv2-v2/out_1096.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/B3E95D55-FEBB-4341-A41E-9A950A702CBF.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018C-UL2018_MiniAODv2-v2/out_1097.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/86E38C9F-DD48-7647-B388-9EF7F8C21868.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018C-UL2018_MiniAODv2-v2/out_1090.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/FCD8B090-CD19-5842-9323-DCA6928C0AD6.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018C-UL2018_MiniAODv2-v2/out_1091.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/5091C07E-2423-D240-B7A6-7467C9BF4F6A.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018C-UL2018_MiniAODv2-v2/out_1092.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/8255F441-E18A-134F-AC94-EDCA4068FFE5.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018C-UL2018_MiniAODv2-v2/out_1093.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/65E418DC-EC60-0440-B86E-72ED74BD5FA8.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018C-UL2018_MiniAODv2-v2/out_1155.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/67C93B35-C768-D641-944E-1E561AF5BE75.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018C-UL2018_MiniAODv2-v2/out_1154.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/1A7C0249-8799-0D45-8058-04993D6EB5AF.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018C-UL2018_MiniAODv2-v2/out_1157.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/75282020-C4DD-8E47-92C8-C137A1DE8CF9.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018C-UL2018_MiniAODv2-v2/out_1156.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/E7E53BCF-A07E-0A43-BDDC-1690EFFF36EC.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018C-UL2018_MiniAODv2-v2/out_1098.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70001/81CB4153-4F1B-A94F-B0B2-EB79A6E7AE2C.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018C-UL2018_MiniAODv2-v2/out_1150.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70001/D16EE180-5512-A745-B528-491DE21BCA16.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018C-UL2018_MiniAODv2-v2/out_1153.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70001/2A5A123F-6D71-A142-9B2E-4C8A31A2A943.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018C-UL2018_MiniAODv2-v2/out_1152.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/2FF138EB-C12D-7A49-B204-2F667100CB71.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018C-UL2018_MiniAODv2-v2/out_263.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/BD80E968-876A-9048-BBAC-ECFC6D804580.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018C-UL2018_MiniAODv2-v2/out_262.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70002/2829CE62-A3A2-BD4A-AA3C-AF373A604FF2.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018C-UL2018_MiniAODv2-v2/out_261.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70002/9AEA164E-771E-074E-8D67-54ACBA53CB23.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018C-UL2018_MiniAODv2-v2/out_260.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/C07084EE-AC09-3549-941C-06432BD493DA.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018C-UL2018_MiniAODv2-v2/out_267.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/003FE915-8683-D747-8929-76D72B22EF6B.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018C-UL2018_MiniAODv2-v2/out_266.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/70DD4218-6E6F-A44D-8587-73E92A857001.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018C-UL2018_MiniAODv2-v2/out_265.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/A68F69B9-62F1-AE40-A155-FA73E77B35BB.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018C-UL2018_MiniAODv2-v2/out_264.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/01F322E7-7325-224D-B8E6-9F56111871E8.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018C-UL2018_MiniAODv2-v2/out_1121.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/DDED7770-B541-9145-BA26-C8E43264906A.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018C-UL2018_MiniAODv2-v2/out_269.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/C641285E-10DE-1645-BB90-19208A4D0380.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018C-UL2018_MiniAODv2-v2/out_268.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/AB24E8E4-FC04-BC4A-8FA2-020D03F3811A.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018C-UL2018_MiniAODv2-v2/out_478.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/E4296438-C613-5044-A945-0893FE71D287.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018C-UL2018_MiniAODv2-v2/out_59.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/3941456C-EE9D-2049-BE99-BC1B209BFDBD.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018C-UL2018_MiniAODv2-v2/out_58.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/2574FC1A-941F-F748-B460-6BF0AD1D73C1.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018C-UL2018_MiniAODv2-v2/out_1215.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/76EE3E11-8639-AA4A-8EFB-02F735605407.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018C-UL2018_MiniAODv2-v2/out_55.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/4168202F-19AB-7444-B06E-FFD0D29F0557.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018C-UL2018_MiniAODv2-v2/out_54.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/6C2145DE-58CA-254E-B25D-8329868D6D9A.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018C-UL2018_MiniAODv2-v2/out_57.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/D111FCB3-61C7-0548-B2C0-7873AEA800A0.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018C-UL2018_MiniAODv2-v2/out_56.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70001/CD43B5DE-07B0-9346-A349-552D4ED29218.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018C-UL2018_MiniAODv2-v2/out_51.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70002/3AF6C090-A376-A044-A659-A7AA3AD17A1A.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018C-UL2018_MiniAODv2-v2/out_258.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/2E2AEEEC-7D47-AA46-9EC3-E95A70F1A600.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018C-UL2018_MiniAODv2-v2/out_53.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/C251322A-B035-F14C-A1EF-BF831C8866B8.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018C-UL2018_MiniAODv2-v2/out_52.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/B9053988-E76B-E843-99E4-3BD24AC13211.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018C-UL2018_MiniAODv2-v2/out_537.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/5117A058-26F3-AD43-B609-1234CBA1F75B.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018C-UL2018_MiniAODv2-v2/out_536.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/15F159D9-CB52-344B-BDE3-5399CB1FB37C.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018C-UL2018_MiniAODv2-v2/out_535.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/EEBD5F44-E97A-764D-B83B-CC49016E7B43.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018C-UL2018_MiniAODv2-v2/out_63.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/DA754C91-EF96-2645-A3BC-64A05FE60417.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018C-UL2018_MiniAODv2-v2/out_533.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/2D089D6D-A5AC-684B-A6E4-03D47588202E.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018C-UL2018_MiniAODv2-v2/out_532.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/AEE9C19B-DCAA-AC43-8C7C-574CF36BB93D.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018C-UL2018_MiniAODv2-v2/out_531.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/4354D1E9-B06B-AD4B-8CD2-C105DA942BBE.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018C-UL2018_MiniAODv2-v2/out_530.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70002/CEF5A394-F9DF-994F-8570-C0176714994E.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018C-UL2018_MiniAODv2-v2/out_598.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/7B5D5050-358F-7243-AF79-813B536EFF23.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018C-UL2018_MiniAODv2-v2/out_152.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/07893445-1B0E-E44C-8D15-724CAC4D429E.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018C-UL2018_MiniAODv2-v2/out_539.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/C2DB2D8C-33BE-7649-91B9-6F082B0B35BE.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018C-UL2018_MiniAODv2-v2/out_538.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/B20B493A-A8EA-0748-9A7B-2DF50A42BAF6.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018C-UL2018_MiniAODv2-v2/out_775.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/14CC78D7-AA89-7A45-A787-74C7D02D293F.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018C-UL2018_MiniAODv2-v2/out_988.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/0CE9E888-EFD9-4145-9EA4-5CAB6B9625D5.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018C-UL2018_MiniAODv2-v2/out_989.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70001/37618D52-8EAF-2045-B3DF-50F08199D83C.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018C-UL2018_MiniAODv2-v2/out_1171.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/66B83052-1795-6448-BFFC-92C845408020.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018C-UL2018_MiniAODv2-v2/out_774.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/8873BF24-7D19-8D40-A9CC-D541BEC8829C.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018C-UL2018_MiniAODv2-v2/out_982.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/991D2014-40BA-D94F-A6E9-7A2DCE2E17E6.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018C-UL2018_MiniAODv2-v2/out_983.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/53ADF2BE-D104-4C47-A988-CA5366DA27B9.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018C-UL2018_MiniAODv2-v2/out_980.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/6BE1DB93-49BF-5442-9151-D5892F451070.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018C-UL2018_MiniAODv2-v2/out_981.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/39678388-FAB5-9C4E-B5DE-D3FFDFD63A86.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018C-UL2018_MiniAODv2-v2/out_986.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/CB52E547-DD19-114D-869C-297D0933AADE.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018C-UL2018_MiniAODv2-v2/out_987.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/9D4EEACC-F381-6244-9A6D-B30949598B09.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018C-UL2018_MiniAODv2-v2/out_984.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70001/1013253B-3F0F-B043-85D8-D26EA5C50BC1.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018C-UL2018_MiniAODv2-v2/out_50.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/AADB323F-306C-1543-A8C3-503BB51566E8.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018C-UL2018_MiniAODv2-v2/out_115.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70002/CC3EB895-38C5-9049-AF5D-839082A0BC84.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018C-UL2018_MiniAODv2-v2/out_252.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70001/1D5A2B90-9948-0241-B935-ED70F0959279.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018C-UL2018_MiniAODv2-v2/out_117.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/8CEAA78D-2B0B-DE45-BB0C-458E96FB3489.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018C-UL2018_MiniAODv2-v2/out_776.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/3C37F359-73F0-ED47-A64F-0D1F0F414FA4.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018C-UL2018_MiniAODv2-v2/out_111.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/B97F2289-E44F-9E4A-B260-3755DBE8AA30.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018C-UL2018_MiniAODv2-v2/out_110.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/E1C6C276-71EF-5E4D-B5C5-9BE0D8950D4C.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018C-UL2018_MiniAODv2-v2/out_113.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/0D56C67B-047B-024A-BCDC-F619778E44E1.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018C-UL2018_MiniAODv2-v2/out_69.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/03832509-620C-F74D-B4D2-BCAFD8F57295.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018C-UL2018_MiniAODv2-v2/out_677.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70001/08B50CAC-55A4-5C42-A5AF-38CAD4051259.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018C-UL2018_MiniAODv2-v2/out_771.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/15036712-3733-534C-9F2D-71D3A4979D95.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018C-UL2018_MiniAODv2-v2/out_5.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70001/04018494-5352-AF41-90D3-3B6DC7473A57.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018C-UL2018_MiniAODv2-v2/out_119.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70001/AD2171F0-DA2F-554C-B685-4942B2E67651.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018C-UL2018_MiniAODv2-v2/out_118.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70001/36DC36BA-07E4-2747-B5EF-05FE906E8BC3.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018C-UL2018_MiniAODv2-v2/out_770.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/E957C93B-C870-0045-A5D8-DD189DA3BE15.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018C-UL2018_MiniAODv2-v2/out_773.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/12445C1C-FD76-894F-BC93-F8C740DAB859.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018C-UL2018_MiniAODv2-v2/out_772.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/448870E1-DE29-9F40-8B41-F25F45152C5F.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018C-UL2018_MiniAODv2-v2/out_1001.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/CAF7BBC0-E304-7941-9A23-DBF03C0D8436.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018C-UL2018_MiniAODv2-v2/out_429.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/B9A63AB3-723A-1641-8422-7E84EF3AA2B3.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018C-UL2018_MiniAODv2-v2/out_428.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/8A3745C4-73EF-AA4F-A052-D975D0496283.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018C-UL2018_MiniAODv2-v2/out_534.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/B75D5ECE-B8C9-284A-A982-FA48B1B392EF.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018C-UL2018_MiniAODv2-v2/out_919.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/1D9794ED-1D7B-9E4D-ADEF-307E0FE41B0B.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018C-UL2018_MiniAODv2-v2/out_918.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/F3553648-3A28-7D41-A920-6EAD832B2FA0.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018C-UL2018_MiniAODv2-v2/out_915.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/1D7DADDF-1010-3E41-BEF5-8AA1FE350E1F.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018C-UL2018_MiniAODv2-v2/out_914.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/DA1AC962-7035-2C4A-9014-4AA4B5F3AEF6.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018C-UL2018_MiniAODv2-v2/out_917.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/E451E593-DF15-5A42-9B7E-E315CF73914A.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018C-UL2018_MiniAODv2-v2/out_916.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/9428A34B-137D-3343-BDF4-859F8B8B5C04.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018C-UL2018_MiniAODv2-v2/out_911.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/EEB9F88A-3107-834D-8EEC-B023956AC0B1.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018C-UL2018_MiniAODv2-v2/out_910.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/CBA9DECD-D395-7C40-9E60-E3D61503E24D.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018C-UL2018_MiniAODv2-v2/out_913.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/31C96604-2E41-0247-9A09-F1B511D48345.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018C-UL2018_MiniAODv2-v2/out_912.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/9EEA7DC5-CE60-8647-B618-D186CCE52FB0.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018C-UL2018_MiniAODv2-v2/out_308.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/1BA52A5E-F45F-C840-BB99-BA7A76E8722A.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018C-UL2018_MiniAODv2-v2/out_309.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/6EE73FE6-0B01-9D47-AECB-CFD7453CC024.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018C-UL2018_MiniAODv2-v2/out_855.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/2FB3D325-0325-1742-B4DA-5393E83EF856.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018C-UL2018_MiniAODv2-v2/out_300.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/DB3D4A64-4004-1B42-B4F4-7F265AC8DF06.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018C-UL2018_MiniAODv2-v2/out_301.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/D8189B09-D465-984B-927B-57DE6DF8D908.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018C-UL2018_MiniAODv2-v2/out_302.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/5D3B1061-A1D1-154E-9BEF-618B7604C999.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018C-UL2018_MiniAODv2-v2/out_303.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/21A4090F-9D96-7640-97DE-75CBD38D1BF2.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018C-UL2018_MiniAODv2-v2/out_304.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/42AE698B-7AB1-184E-99F7-3EFB41A85438.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018C-UL2018_MiniAODv2-v2/out_305.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/E1615FD1-8F55-B448-8606-0FAB26EFF446.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018C-UL2018_MiniAODv2-v2/out_306.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/4C4442D4-E8C1-AC4E-872B-7A52917A580C.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018C-UL2018_MiniAODv2-v2/out_307.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/09D05A17-23A5-D646-AA89-24383B3C2218.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018C-UL2018_MiniAODv2-v2/out_895.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/4AA18061-6D1E-7949-B8B7-216A04133841.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018C-UL2018_MiniAODv2-v2/out_825.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/055612C0-C44D-6240-A799-015B56D891F8.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018C-UL2018_MiniAODv2-v2/out_665.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/58F2738A-69AA-774F-8276-E905DD4F3014.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018C-UL2018_MiniAODv2-v2/out_824.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/FE172196-28C4-B54D-81D0-C3755A16F8E8.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018C-UL2018_MiniAODv2-v2/out_649.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/1B05BA51-FE22-9C4E-ABA0-E0802773E66D.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018C-UL2018_MiniAODv2-v2/out_847.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/84D452F2-9B7C-C346-A729-9915533638DC.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018C-UL2018_MiniAODv2-v2/out_846.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/D1231B9D-FB2E-E042-B22C-DD91706B4650.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018C-UL2018_MiniAODv2-v2/out_845.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/16F7AE01-A8F0-784C-A6DE-FC814E29F495.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018C-UL2018_MiniAODv2-v2/out_844.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/0609C06A-2375-7649-B9FC-33BF21231856.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018C-UL2018_MiniAODv2-v2/out_843.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/98938A70-313F-E746-80BB-AFC7CFAF48EA.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018C-UL2018_MiniAODv2-v2/out_842.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/39E53AB1-3BE6-8A40-BBD4-525AD839A4AE.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018C-UL2018_MiniAODv2-v2/out_841.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/51DBAF14-4514-BE49-AAD9-515A4E03F531.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018C-UL2018_MiniAODv2-v2/out_840.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70001/CF3A0377-42BE-8946-8A7F-2EDBA3F323A0.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018C-UL2018_MiniAODv2-v2/out_1151.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/CCF1DC0D-0964-B648-AD09-85267008B02C.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018C-UL2018_MiniAODv2-v2/out_821.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/39285033-6C79-CB47-8E1E-BD05C3C967CE.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018C-UL2018_MiniAODv2-v2/out_853.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/5FBCB6E1-19AC-7944-B46D-B8AA8244AB9C.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018C-UL2018_MiniAODv2-v2/out_1126.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/7A9B24BC-6029-B348-B49B-6BF341CA3BA5.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018C-UL2018_MiniAODv2-v2/out_848.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/6C3BC962-38A2-864A-9B42-9D1981B2A88C.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018C-UL2018_MiniAODv2-v2/out_823.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70001/866CD9FD-4DF2-DE4D-B9A0-02011104E3BF.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018C-UL2018_MiniAODv2-v2/out_1209.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/1F98B58B-9F6B-854E-883C-22995367B352.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018C-UL2018_MiniAODv2-v2/out_822.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/40C90571-5283-9547-BD19-3A39E0883E3E.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018C-UL2018_MiniAODv2-v2/out_1127.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/8EE039F0-45CB-D841-96CB-D5270346B511.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018C-UL2018_MiniAODv2-v2/out_240.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70001/EA11B035-32CD-9642-AA40-05681C06C95E.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018C-UL2018_MiniAODv2-v2/out_1208.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/CD0EDFB9-A91C-F445-BC20-CBB4896702C7.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018C-UL2018_MiniAODv2-v2/out_485.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/D91E3DEE-98DF-344A-B2C4-E6C5BEFCA6FC.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018C-UL2018_MiniAODv2-v2/out_390.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/DC8BE2EF-7A22-9446-8FCB-A4DA9D7F4B73.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018C-UL2018_MiniAODv2-v2/out_1038.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70001/62E7DADE-F460-9C49-9D2A-6AE522C2132F.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018C-UL2018_MiniAODv2-v2/out_732.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/D760290F-0BC0-0848-AFCA-77E1EEB7949E.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018C-UL2018_MiniAODv2-v2/out_162.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/5CAC8BD0-11A3-004B-A378-FB0F6D1A060C.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018C-UL2018_MiniAODv2-v2/out_1166.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/4089F857-5DD4-AF4D-91D3-CC5565BC7455.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018C-UL2018_MiniAODv2-v2/out_391.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70001/801BC48C-7B69-F84E-933C-41DCEFE73F19.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018C-UL2018_MiniAODv2-v2/out_753.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70001/7A71C504-073B-3146-8A08-6805ECC74946.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018C-UL2018_MiniAODv2-v2/out_752.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70001/F07189C9-BBF7-CE4E-AC81-A99B425C61FD.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018C-UL2018_MiniAODv2-v2/out_751.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70001/FB438111-6237-7141-934C-D850634EA9C0.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018C-UL2018_MiniAODv2-v2/out_750.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70001/C62D3CFD-9E08-4349-99D0-A70A9959773D.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018C-UL2018_MiniAODv2-v2/out_757.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70001/6002454D-B75D-0D4E-A542-FEA83B178E49.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018C-UL2018_MiniAODv2-v2/out_756.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70001/BCA862DA-23D0-B649-876B-594448653D6B.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018C-UL2018_MiniAODv2-v2/out_755.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70001/7F1D20EB-8BE1-CE49-948F-84C9EEBF9839.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018C-UL2018_MiniAODv2-v2/out_754.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/2C374441-C27B-7544-8032-721104986962.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018C-UL2018_MiniAODv2-v2/out_560.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70001/E3D78381-B829-454F-BB1F-157826955629.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018C-UL2018_MiniAODv2-v2/out_561.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70001/6E91BB96-B1CB-BF41-B2C0-15A0B8624F73.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018C-UL2018_MiniAODv2-v2/out_759.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70001/A01BA1E9-C55E-5C41-A146-B454460D58E7.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018C-UL2018_MiniAODv2-v2/out_758.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/08431921-B81E-F940-9954-C28FECE02B64.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018C-UL2018_MiniAODv2-v2/out_564.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/2ED56A01-1D2D-744C-8F68-DE36B5FD6D40.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018C-UL2018_MiniAODv2-v2/out_565.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/3F251D6B-F72F-3C4E-BDA5-994470F566B7.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018C-UL2018_MiniAODv2-v2/out_566.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/731A7D94-7A7C-F84E-9671-98D157093A60.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018C-UL2018_MiniAODv2-v2/out_701.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70001/5A2A0F44-8E56-254F-B954-24E736C90F76.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018C-UL2018_MiniAODv2-v2/out_506.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/6ACF66AA-291A-244B-ADB4-102580319758.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018C-UL2018_MiniAODv2-v2/out_229.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70001/54AE8C0F-0579-B948-B955-F72EF8E00970.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018C-UL2018_MiniAODv2-v2/out_507.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/E425B85E-FC63-4041-8252-D54241DC0C75.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018C-UL2018_MiniAODv2-v2/out_227.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/E2A4BDF2-77AD-7649-9DA4-C7973F9BFC6E.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018C-UL2018_MiniAODv2-v2/out_226.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/B0551FC0-ADA7-A545-A493-01E401753BAF.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018C-UL2018_MiniAODv2-v2/out_225.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/4193E9EC-B9C4-5846-BF80-8F39EB87D576.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018C-UL2018_MiniAODv2-v2/out_985.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/285BDFF7-A72B-F949-AA9D-6C8E05EC4F9E.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018C-UL2018_MiniAODv2-v2/out_223.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/D6468C98-A324-9A4E-BBB9-855A5BB47F08.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018C-UL2018_MiniAODv2-v2/out_504.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/A69F9BE8-588A-4141-9FEF-CE6F4ABDA839.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018C-UL2018_MiniAODv2-v2/out_221.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/F42E368A-1A60-3A4C-BE83-F0F7D6D33D38.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018C-UL2018_MiniAODv2-v2/out_220.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/B8B58C89-E067-C044-99F8-71ADFC971740.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018C-UL2018_MiniAODv2-v2/out_1025.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/F1DB7860-6254-024F-8B36-B3BC56CAA283.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018C-UL2018_MiniAODv2-v2/out_1024.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/110BE18E-FDA1-3543-90D9-769E7A12AA76.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018C-UL2018_MiniAODv2-v2/out_1027.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/412A3BE0-28D9-304A-BD9F-AC0A652F9F2D.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018C-UL2018_MiniAODv2-v2/out_1031.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/91F4B2C6-2486-034E-9250-8A29233D053A.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018C-UL2018_MiniAODv2-v2/out_1021.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/27BA2C10-02EE-0440-B4D3-66BE4D3BA5A7.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018C-UL2018_MiniAODv2-v2/out_1020.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/F051F7F0-4D80-F04D-9A4D-36538183428C.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018C-UL2018_MiniAODv2-v2/out_1023.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/CCE98BFF-BC0A-C345-9D4E-8B07B1931587.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018C-UL2018_MiniAODv2-v2/out_1189.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70001/36C15A8A-D590-B34A-8A1F-4848608A267E.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018C-UL2018_MiniAODv2-v2/out_1186.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70001/491F207D-FD1E-4549-971F-DBCF424FCE07.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018C-UL2018_MiniAODv2-v2/out_1187.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70001/2926D219-A0F3-1649-9764-B3F3C854CA03.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018C-UL2018_MiniAODv2-v2/out_1184.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70001/E6B01F32-4D78-D241-9770-A604E9BD9288.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018C-UL2018_MiniAODv2-v2/out_1185.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70001/1AFABC2E-D807-BC4C-9DAA-49E4533AA324.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018C-UL2018_MiniAODv2-v2/out_1182.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70001/CDD8F59B-63C4-1B46-AFF5-1EAE6237132A.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018C-UL2018_MiniAODv2-v2/out_1183.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/14CD3BDC-BB6B-A546-8598-D4393E1EDF1C.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018C-UL2018_MiniAODv2-v2/out_1180.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70001/D64D967E-129C-A24B-AE15-1EE6724AB405.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018C-UL2018_MiniAODv2-v2/out_1181.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/6E4840C6-0A0F-C848-A675-49749C53E8D6.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018C-UL2018_MiniAODv2-v2/out_726.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/74EA9E7D-0D1B-D64A-A4B0-BCF4177DCAC9.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018C-UL2018_MiniAODv2-v2/out_727.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/CC631BEC-F19E-5F4A-B15E-7D363DDB4524.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018C-UL2018_MiniAODv2-v2/out_724.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/DE725A19-2B39-DD4F-8AF7-65FF5309AEF6.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018C-UL2018_MiniAODv2-v2/out_89.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/FE0DD9CB-B922-5C49-8C0A-0828CD2D9F67.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018C-UL2018_MiniAODv2-v2/out_568.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/FECFB505-C8FF-314D-A5DF-D4CC18CE2907.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018C-UL2018_MiniAODv2-v2/out_1034.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/D51CF148-6F32-FE47-824A-5DE8F77CF00A.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018C-UL2018_MiniAODv2-v2/out_426.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/AEFD6D26-7A17-5C47-9938-D9A27B8A68C7.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018C-UL2018_MiniAODv2-v2/out_721.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/964437DB-4A90-0645-9836-BCDA65318158.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018C-UL2018_MiniAODv2-v2/out_1160.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/A6F3D65A-A1F3-7642-8A09-EB852822E5D2.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018C-UL2018_MiniAODv2-v2/out_1035.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/3557E9BE-0FDB-7249-A807-0E2908E93794.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018C-UL2018_MiniAODv2-v2/out_1167.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/4522D25D-D00A-C44F-8F27-C54E89E92E2E.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018C-UL2018_MiniAODv2-v2/out_728.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/BAF5FFC5-95BF-CF44-BC69-DBE16EC0FF1A.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018C-UL2018_MiniAODv2-v2/out_729.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70002/F5F4B603-BD41-4B48-B76A-C3926CA52E1E.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018C-UL2018_MiniAODv2-v2/out_605.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/A08089A7-9C4C-6949-AE98-91A732626B4B.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018C-UL2018_MiniAODv2-v2/out_604.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/C743C770-25FD-794C-A950-E42F8FD7E927.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018C-UL2018_MiniAODv2-v2/out_607.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/0D0372A0-8F54-D644-9DA0-09DD2571D685.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018C-UL2018_MiniAODv2-v2/out_606.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70002/7EE842A2-57D8-CD4C-A204-1356CE5C6DB9.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018C-UL2018_MiniAODv2-v2/out_601.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/4415F3EC-BB6F-3D46-94E9-4FDC4FB01C41.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018C-UL2018_MiniAODv2-v2/out_600.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70001/C927FCF4-A93C-0C45-990B-F22CC1F838CC.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018C-UL2018_MiniAODv2-v2/out_1162.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/D75B6E57-D95F-7745-A714-989E51582CAA.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018C-UL2018_MiniAODv2-v2/out_602.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70001/F73468E4-728A-EB47-9811-C48BF6F817A1.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018C-UL2018_MiniAODv2-v2/out_1205.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70001/AA312D05-56A4-834E-9207-BAB2A4CF1D00.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018C-UL2018_MiniAODv2-v2/out_158.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70001/195314B5-E4AE-1B48-9AF7-B5D14C425E12.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018C-UL2018_MiniAODv2-v2/out_1207.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/B96FB99B-3024-1F46-B9B5-A380917FCEC0.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018C-UL2018_MiniAODv2-v2/out_112.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/CEBF6EDA-6606-4341-A00D-F82A8AF58CFC.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018C-UL2018_MiniAODv2-v2/out_1168.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/311FE0C4-2552-6F43-8839-2509BBBBA2F4.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018C-UL2018_MiniAODv2-v2/out_1200.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/BE883381-DA11-594E-BD43-B5C620FCAF3D.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018C-UL2018_MiniAODv2-v2/out_1203.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/D7739588-3713-9945-86FF-4AFFBD2F04F0.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018C-UL2018_MiniAODv2-v2/out_1202.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/ED05F728-66F1-DA4C-8154-F71F68B69373.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018C-UL2018_MiniAODv2-v2/out_1204.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/A9FA970B-F6FF-7A49-B4EE-FACE88D1E423.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018C-UL2018_MiniAODv2-v2/out_1211.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/06DAF11D-9FE6-4448-941E-81909F107EEB.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018C-UL2018_MiniAODv2-v2/out_976.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/F896E22B-F538-6A45-B7CD-F3CD44104F51.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018C-UL2018_MiniAODv2-v2/out_1161.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70001/6535D5C7-D142-1647-B44F-57AB972F6EC7.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018C-UL2018_MiniAODv2-v2/out_469.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/61E50889-39BF-B642-BB36-45B8B5CEA40C.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018C-UL2018_MiniAODv2-v2/out_935.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/DD33816D-1C71-A246-A27E-8BA1C41EF7CD.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018C-UL2018_MiniAODv2-v2/out_399.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/95DF413B-086B-5443-9D35-8CDAEE09FA03.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018C-UL2018_MiniAODv2-v2/out_1217.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/FAF3564D-BF9A-B846-B11D-123D5DC0E999.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018C-UL2018_MiniAODv2-v2/out_959.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/8424F8A6-2753-524F-B184-7DC07DFCB149.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018C-UL2018_MiniAODv2-v2/out_958.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/BFA6A0DA-38BD-D844-B811-18046F27E674.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018C-UL2018_MiniAODv2-v2/out_398.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/A6A2FFC3-1E7B-9149-B1A6-7C6CB1A51E73.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018C-UL2018_MiniAODv2-v2/out_1214.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70001/7F64DBEF-C39F-EC44-A9A9-2B3530E73DEB.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018C-UL2018_MiniAODv2-v2/out_48.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70001/9E94FDEC-B812-0449-B6A0-D4E96421296E.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018C-UL2018_MiniAODv2-v2/out_49.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/08D2B133-1A80-D241-AA2D-6272BA751BBF.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018C-UL2018_MiniAODv2-v2/out_951.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/3235616B-BDFF-9C40-B509-59BFDD85A84D.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018C-UL2018_MiniAODv2-v2/out_950.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/29040451-48B1-7343-A073-B1E513044527.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018C-UL2018_MiniAODv2-v2/out_953.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/BC9FBED0-7D5F-1D41-91E2-C381D1BB6A17.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018C-UL2018_MiniAODv2-v2/out_952.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/CB17BF2C-986B-294E-9D57-2E3E3B02A226.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018C-UL2018_MiniAODv2-v2/out_42.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/F518EC2A-DFAA-6C45-987D-6DC2934D8C33.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018C-UL2018_MiniAODv2-v2/out_954.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/2C4F2E2B-962D-AE4F-9BD3-B9F14D4AC2FE.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018C-UL2018_MiniAODv2-v2/out_957.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/6C877D7A-745E-E143-9832-5EC22A6457BB.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018C-UL2018_MiniAODv2-v2/out_956.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/1DA66FB4-AD86-B040-BE01-E37C5DCD5140.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018C-UL2018_MiniAODv2-v2/out_1111.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/6A5F9579-D9F1-2547-8B23-061BAF37DF55.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018C-UL2018_MiniAODv2-v2/out_1110.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/A5F6C3AA-BC18-8C4C-A1BB-80889D2AD50A.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018C-UL2018_MiniAODv2-v2/out_1113.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/FB0B9B2A-F572-514F-B5CC-EF21E81D6D55.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018C-UL2018_MiniAODv2-v2/out_1112.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70001/39112B93-5686-E244-91E1-52CCC262A4CA.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018C-UL2018_MiniAODv2-v2/out_1115.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/61568136-2137-E245-8C68-CFAD84BA30BB.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018C-UL2018_MiniAODv2-v2/out_1114.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70001/2965E761-67B5-514F-9596-B81F30CDF94D.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018C-UL2018_MiniAODv2-v2/out_1117.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70001/AF768659-7B64-9A48-9F44-D9FE584514D9.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018C-UL2018_MiniAODv2-v2/out_1116.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/FA9DDE1D-8B79-C846-950B-9BCC19B78080.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018C-UL2018_MiniAODv2-v2/out_620.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/BAC142B8-71AD-BB4F-8F44-FFF12A41D119.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018C-UL2018_MiniAODv2-v2/out_608.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/04E80FB8-7F3F-C349-9C05-3A94B967451A.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018C-UL2018_MiniAODv2-v2/out_467.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70001/62F1EAB4-52A9-174D-B716-34150F86183B.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018C-UL2018_MiniAODv2-v2/out_1188.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/620DD078-D498-294F-889D-87A1CC7BB8B2.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018C-UL2018_MiniAODv2-v2/out_1219.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70001/369557D0-7687-EE48-8446-DE7711D3FF48.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018C-UL2018_MiniAODv2-v2/out_1206.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/0CFAB0CE-059D-B949-AB96-B3315FEA575A.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018C-UL2018_MiniAODv2-v2/out_460.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/1B07B538-5B14-2045-95EB-A21AA950486C.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018C-UL2018_MiniAODv2-v2/out_1163.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/D774CF81-0F62-D040-9161-31827012E56A.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018C-UL2018_MiniAODv2-v2/out_489.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/E3610DF1-91D8-6846-B26B-741D6B42C8FE.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018C-UL2018_MiniAODv2-v2/out_488.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/36C17296-4B44-C54F-AF46-5AC983F1A213.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018C-UL2018_MiniAODv2-v2/out_487.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/0A9C5B8A-336F-1F4B-B038-2C2BCEA8E3C9.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018C-UL2018_MiniAODv2-v2/out_486.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/A6A73511-83AE-EC49-9CEC-58751B6D4D71.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018C-UL2018_MiniAODv2-v2/out_869.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70001/AE29ED46-8FBE-A04C-A738-979A44E00421.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018C-UL2018_MiniAODv2-v2/out_484.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70001/949C76F8-FF3B-D84C-9102-F9427C8F8B4E.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018C-UL2018_MiniAODv2-v2/out_483.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70001/5BF6FFA2-8464-9D44-BD87-13F76F2C920C.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018C-UL2018_MiniAODv2-v2/out_482.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/050462F8-D2D6-864A-BF63-26CB41C64249.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018C-UL2018_MiniAODv2-v2/out_481.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/E13C0AE9-5CC8-004F-9665-1EA61A0F114E.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018C-UL2018_MiniAODv2-v2/out_480.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/3FE84C5C-EA70-0B4B-A340-0E5C92FFA5C6.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018C-UL2018_MiniAODv2-v2/out_509.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/9EDEB902-E0EA-EE40-93FE-FF1332EDFA86.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018C-UL2018_MiniAODv2-v2/out_955.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/7A4E785C-3930-E942-93E0-476896A6954E.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018C-UL2018_MiniAODv2-v2/out_1201.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/6516A5CC-A870-DA4B-B6F5-0F4EA29A88E0.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018C-UL2018_MiniAODv2-v2/out_472.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/D6C7939B-63B8-5043-A628-667C4AC9AA8C.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018C-UL2018_MiniAODv2-v2/out_473.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/B7825C78-96FC-E449-A281-0F0F54AB4814.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018C-UL2018_MiniAODv2-v2/out_470.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/3089FE7E-A6A4-654D-B5F5-EFDE8E1B3E3C.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018C-UL2018_MiniAODv2-v2/out_471.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/DD227FF3-0926-B047-8D73-71D1146C5144.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018C-UL2018_MiniAODv2-v2/out_476.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/AEE2B1F2-8664-2445-B013-14F723E9CCA6.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018C-UL2018_MiniAODv2-v2/out_477.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/50CB54B1-D005-AA4C-85C2-3DE14F41C29B.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018C-UL2018_MiniAODv2-v2/out_474.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/FF9094B5-957D-4247-9526-F53D51FBBAD8.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018C-UL2018_MiniAODv2-v2/out_475.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/ABA1DF18-F30D-EF41-9DF1-87C19B00E93D.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018C-UL2018_MiniAODv2-v2/out_46.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/SingleMuon/MINIAOD/UL2018_MiniAODv2-v2/70000/D0B9E6AE-AB95-834B-859E-AE71CCB8CC29.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/SingleMuon_Run2018C-UL2018_MiniAODv2-v2/out_479.root -a '-o ./ -d --year UL2018C'"
queue
