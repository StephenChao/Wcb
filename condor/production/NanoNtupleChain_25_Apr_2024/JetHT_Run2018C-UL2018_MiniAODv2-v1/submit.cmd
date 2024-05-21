universe=Vanilla
+ProjectName = "cms.org"
requirements = (OpSysAndVer =?= "CentOS7")
RequestMemory = 2048
RequestCpus = 1
executable=/home/legao/Wcb/condor/production/NanoNtupleChain_25_Apr_2024/JetHT_Run2018C-UL2018_MiniAODv2-v1/exe_CUSTNANO_UL18_MINIAODv2_DATA.sh
transfer_executable=True
transfer_input_files=/home/legao/Wcb/condor/x509up_u14433,/home/legao/Wcb/condor/Scripts/fetchFiles.py
transfer_output_files=""
log=/home/legao/Wcb/condor/production/NanoNtupleChain_25_Apr_2024/JetHT_Run2018C-UL2018_MiniAODv2-v1/log//$(Cluster).log
output=/home/legao/Wcb/condor/production/NanoNtupleChain_25_Apr_2024/JetHT_Run2018C-UL2018_MiniAODv2-v1/log//$(Cluster).out
error=/home/legao/Wcb/condor/production/NanoNtupleChain_25_Apr_2024/JetHT_Run2018C-UL2018_MiniAODv2-v1/log//$(Cluster).err
notification=Never
should_transfer_files = YES
when_to_transfer_output = ON_EXIT
x509userproxy=x509up_u14433
+MaxRuntime=150000
on_exit_remove        = (ExitBySignal == False) && (ExitCode == 0)
on_exit_hold          = (ExitBySignal == True) || (ExitCode != 0)
on_exit_hold_reason   = strcat("Job held by ON_EXIT_HOLD due to ", ifThenElse((ExitBySignal == True), strcat("exit signal ", ExitSignal), strcat("exit code ", ExitCode)), ".")
periodic_release      = (NumJobStarts < 3) && ((CurrentTime - EnteredCurrentStatus) > 60*60)
arguments="-f /store/data/Run2018C/JetHT/MINIAOD/UL2018_MiniAODv2-v1/50000/F8C02175-9F28-8D4A-A9FD-A8FDB9B24376.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018C-UL2018_MiniAODv2-v1/out_723.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/JetHT/MINIAOD/UL2018_MiniAODv2-v1/40000/C6C97564-1290-794E-BF35-1AA381A61904.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018C-UL2018_MiniAODv2-v1/out_198.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/JetHT/MINIAOD/UL2018_MiniAODv2-v1/50000/433DC1CD-22FF-8440-81AC-974A7380EB5D.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018C-UL2018_MiniAODv2-v1/out_669.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/JetHT/MINIAOD/UL2018_MiniAODv2-v1/50000/F5237DCB-EC3D-8E4C-9922-5B9FADA166C3.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018C-UL2018_MiniAODv2-v1/out_668.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/JetHT/MINIAOD/UL2018_MiniAODv2-v1/40000/15FD3C08-9025-8D4B-8891-624C79D401B2.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018C-UL2018_MiniAODv2-v1/out_344.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/JetHT/MINIAOD/UL2018_MiniAODv2-v1/40000/CE93005F-5A2C-694A-9B38-0E40FFBBC214.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018C-UL2018_MiniAODv2-v1/out_819.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/JetHT/MINIAOD/UL2018_MiniAODv2-v1/50000/CA7346A7-71C3-C94B-8673-42AEA493F866.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018C-UL2018_MiniAODv2-v1/out_346.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/JetHT/MINIAOD/UL2018_MiniAODv2-v1/50000/BD52380E-42A5-8C4E-B74B-202DC8D8E3B5.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018C-UL2018_MiniAODv2-v1/out_347.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/JetHT/MINIAOD/UL2018_MiniAODv2-v1/40000/4C63618F-E2F1-2245-849D-9DC97675D68C.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018C-UL2018_MiniAODv2-v1/out_340.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/JetHT/MINIAOD/UL2018_MiniAODv2-v1/40000/A47C7196-C5CF-C347-BEF1-B52BFBA25A63.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018C-UL2018_MiniAODv2-v1/out_341.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/JetHT/MINIAOD/UL2018_MiniAODv2-v1/40000/FC322B1E-403C-B848-A205-393237BF2838.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018C-UL2018_MiniAODv2-v1/out_342.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/JetHT/MINIAOD/UL2018_MiniAODv2-v1/40000/531537B6-2C93-4148-87B1-99CEC4FE503B.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018C-UL2018_MiniAODv2-v1/out_343.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/JetHT/MINIAOD/UL2018_MiniAODv2-v1/40000/2F3F4C5D-8EEC-0E42-80E1-47E80FFD62A6.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018C-UL2018_MiniAODv2-v1/out_810.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/JetHT/MINIAOD/UL2018_MiniAODv2-v1/40000/46661E3C-75A4-E84C-B0E3-8D484B710DE0.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018C-UL2018_MiniAODv2-v1/out_811.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/JetHT/MINIAOD/UL2018_MiniAODv2-v1/40000/0184A215-4024-9B4D-97C6-7469E77932B8.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018C-UL2018_MiniAODv2-v1/out_812.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/JetHT/MINIAOD/UL2018_MiniAODv2-v1/40000/FD98A9D7-3708-8D4D-892E-693AE3502C2A.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018C-UL2018_MiniAODv2-v1/out_813.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/JetHT/MINIAOD/UL2018_MiniAODv2-v1/50000/3FC4CFA8-15CE-9641-86C0-8F21AF8C2B33.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018C-UL2018_MiniAODv2-v1/out_348.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/JetHT/MINIAOD/UL2018_MiniAODv2-v1/50000/C9B4FF12-7665-0241-84CD-D597A441906D.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018C-UL2018_MiniAODv2-v1/out_349.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/JetHT/MINIAOD/UL2018_MiniAODv2-v1/40000/66F0306A-44E5-BE46-9C45-F6847262D984.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018C-UL2018_MiniAODv2-v1/out_816.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/JetHT/MINIAOD/UL2018_MiniAODv2-v1/40000/DB434A14-562D-3A45-99F5-269F8DCAD3C9.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018C-UL2018_MiniAODv2-v1/out_817.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/JetHT/MINIAOD/UL2018_MiniAODv2-v1/50000/8F6331B6-A1B5-4C46-AAEC-6DE65B114C6D.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018C-UL2018_MiniAODv2-v1/out_664.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/JetHT/MINIAOD/UL2018_MiniAODv2-v1/50000/12C7D8AB-44A6-AE42-9DF0-4EA5B3C7F18D.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018C-UL2018_MiniAODv2-v1/out_739.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/JetHT/MINIAOD/UL2018_MiniAODv2-v1/50000/B5102276-BC28-B24F-9EA9-451AA395DF56.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018C-UL2018_MiniAODv2-v1/out_719.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/JetHT/MINIAOD/UL2018_MiniAODv2-v1/50000/E6B78EA8-CB09-2947-A92B-B071144E5D3E.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018C-UL2018_MiniAODv2-v1/out_718.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/JetHT/MINIAOD/UL2018_MiniAODv2-v1/50000/D45FBF3F-DC23-5B41-9CB8-C56B8217D673.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018C-UL2018_MiniAODv2-v1/out_717.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/JetHT/MINIAOD/UL2018_MiniAODv2-v1/50000/79AEE60F-41D7-3C47-97DB-886EC9EFB8F3.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018C-UL2018_MiniAODv2-v1/out_716.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/JetHT/MINIAOD/UL2018_MiniAODv2-v1/50000/7F576DAA-480C-8942-A7C0-C7F8EF6805A4.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018C-UL2018_MiniAODv2-v1/out_715.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/JetHT/MINIAOD/UL2018_MiniAODv2-v1/50000/114117CE-9471-EB4B-8C91-A5F4E1894A7C.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018C-UL2018_MiniAODv2-v1/out_714.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/JetHT/MINIAOD/UL2018_MiniAODv2-v1/50000/BE84A751-934A-2B40-BDAA-A183F45125A2.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018C-UL2018_MiniAODv2-v1/out_713.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/JetHT/MINIAOD/UL2018_MiniAODv2-v1/50000/EB031780-44EC-3345-A38E-CF21DCAD48EA.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018C-UL2018_MiniAODv2-v1/out_712.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/JetHT/MINIAOD/UL2018_MiniAODv2-v1/50000/B0801F76-84E5-034A-A06A-E3AB3C17649A.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018C-UL2018_MiniAODv2-v1/out_711.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/JetHT/MINIAOD/UL2018_MiniAODv2-v1/50000/28D24271-4518-F840-B61B-077981B9BBC4.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018C-UL2018_MiniAODv2-v1/out_710.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/JetHT/MINIAOD/UL2018_MiniAODv2-v1/50000/FBF33568-1D90-DF4B-9E7F-9648E34E047C.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018C-UL2018_MiniAODv2-v1/out_618.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/JetHT/MINIAOD/UL2018_MiniAODv2-v1/50000/23AD108E-D84E-5E4A-B3D4-9EE23332A417.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018C-UL2018_MiniAODv2-v1/out_421.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/JetHT/MINIAOD/UL2018_MiniAODv2-v1/50000/A26DFF1A-9467-C041-B416-44A3DDDB5DB1.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018C-UL2018_MiniAODv2-v1/out_129.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/JetHT/MINIAOD/UL2018_MiniAODv2-v1/50000/AFCE334E-CD6F-A041-9571-96C5FBA2FB6D.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018C-UL2018_MiniAODv2-v1/out_420.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/JetHT/MINIAOD/UL2018_MiniAODv2-v1/50000/FD432BAE-047F-FC4A-80BB-8FDB290F73B1.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018C-UL2018_MiniAODv2-v1/out_423.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/JetHT/MINIAOD/UL2018_MiniAODv2-v1/50000/57F9892C-AABD-5048-8292-15ADB0545EA0.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018C-UL2018_MiniAODv2-v1/out_1069.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/JetHT/MINIAOD/UL2018_MiniAODv2-v1/50000/F39DCA80-A0FD-C145-8FDB-AADD8C38A81D.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018C-UL2018_MiniAODv2-v1/out_1068.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/JetHT/MINIAOD/UL2018_MiniAODv2-v1/50000/3C6E0993-109D-764B-864B-E67F7AA2382A.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018C-UL2018_MiniAODv2-v1/out_422.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/JetHT/MINIAOD/UL2018_MiniAODv2-v1/50000/2CB0C33F-FC5E-2249-AB84-95B165E413BE.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018C-UL2018_MiniAODv2-v1/out_1061.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/JetHT/MINIAOD/UL2018_MiniAODv2-v1/50000/C2E5455C-EC1C-DB4A-9D59-3EA1588A3604.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018C-UL2018_MiniAODv2-v1/out_1060.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/JetHT/MINIAOD/UL2018_MiniAODv2-v1/50000/D94DA33D-94CC-6149-8AAC-6C5CAA0CDB95.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018C-UL2018_MiniAODv2-v1/out_1063.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/JetHT/MINIAOD/UL2018_MiniAODv2-v1/50000/1167ED49-BB0F-0C45-8468-BD513999DFA5.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018C-UL2018_MiniAODv2-v1/out_1062.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/JetHT/MINIAOD/UL2018_MiniAODv2-v1/50000/40992D8F-E682-4A49-A402-D0069156FE98.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018C-UL2018_MiniAODv2-v1/out_1065.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/JetHT/MINIAOD/UL2018_MiniAODv2-v1/50000/99D8BFDD-F407-5F42-B644-8FCCA4C0EAD2.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018C-UL2018_MiniAODv2-v1/out_1064.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/JetHT/MINIAOD/UL2018_MiniAODv2-v1/50000/FFAEB616-1B47-4945-8BCC-16BA57DFC1BE.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018C-UL2018_MiniAODv2-v1/out_1067.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/JetHT/MINIAOD/UL2018_MiniAODv2-v1/50000/65BFC91A-4B84-8D45-ADD3-562EC63691AC.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018C-UL2018_MiniAODv2-v1/out_1066.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/JetHT/MINIAOD/UL2018_MiniAODv2-v1/50000/D2E3ADDE-5061-3C4E-B6AA-54FC9A9C8C47.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018C-UL2018_MiniAODv2-v1/out_671.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/JetHT/MINIAOD/UL2018_MiniAODv2-v1/50000/7C560582-2493-6449-A597-CAED1F709577.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018C-UL2018_MiniAODv2-v1/out_427.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/JetHT/MINIAOD/UL2018_MiniAODv2-v1/40000/B53351F6-84D7-8E47-BF16-23EEB98CA437.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018C-UL2018_MiniAODv2-v1/out_298.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/JetHT/MINIAOD/UL2018_MiniAODv2-v1/40000/636FD525-D4C8-184C-92E7-21DD32BC6F13.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018C-UL2018_MiniAODv2-v1/out_299.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/JetHT/MINIAOD/UL2018_MiniAODv2-v1/50000/CA55C5B8-8287-8C4D-AA0B-2F58D8285D75.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018C-UL2018_MiniAODv2-v1/out_296.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/JetHT/MINIAOD/UL2018_MiniAODv2-v1/50000/3A3B60D6-17E2-3249-80E4-DA0C1C106457.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018C-UL2018_MiniAODv2-v1/out_297.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/JetHT/MINIAOD/UL2018_MiniAODv2-v1/50000/9F44DC1D-662D-CF42-81DD-C079A2960BD9.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018C-UL2018_MiniAODv2-v1/out_294.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/JetHT/MINIAOD/UL2018_MiniAODv2-v1/50000/A1AA1DCC-261C-C740-8D51-844894BAB4ED.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018C-UL2018_MiniAODv2-v1/out_295.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/JetHT/MINIAOD/UL2018_MiniAODv2-v1/50000/E787176A-3E3A-B748-9150-D247E3626FAA.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018C-UL2018_MiniAODv2-v1/out_292.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/JetHT/MINIAOD/UL2018_MiniAODv2-v1/50000/DCAFF006-8D85-504B-BFD7-2DC96189188A.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018C-UL2018_MiniAODv2-v1/out_293.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/JetHT/MINIAOD/UL2018_MiniAODv2-v1/50000/B3B7068C-45E0-C34D-9FC9-9EAC84C5968C.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018C-UL2018_MiniAODv2-v1/out_290.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/JetHT/MINIAOD/UL2018_MiniAODv2-v1/50000/1BE1B26D-349B-3745-8147-1F3B8DA9F193.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018C-UL2018_MiniAODv2-v1/out_291.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/JetHT/MINIAOD/UL2018_MiniAODv2-v1/50000/E47D3C38-B538-EF48-9DB7-D9B30AFE90DB.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018C-UL2018_MiniAODv2-v1/out_591.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/JetHT/MINIAOD/UL2018_MiniAODv2-v1/50000/A71F97DC-325E-BE42-AE09-7EE0070754CC.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018C-UL2018_MiniAODv2-v1/out_590.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/JetHT/MINIAOD/UL2018_MiniAODv2-v1/50000/5EA5B638-1BF1-A044-961A-F20EF54C2191.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018C-UL2018_MiniAODv2-v1/out_593.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/JetHT/MINIAOD/UL2018_MiniAODv2-v1/50000/71515BF9-A180-B148-83E4-43F276DFA54B.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018C-UL2018_MiniAODv2-v1/out_592.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/JetHT/MINIAOD/UL2018_MiniAODv2-v1/50000/B9AB5BDD-6211-2643-A243-BCB8FDF5443D.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018C-UL2018_MiniAODv2-v1/out_595.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/JetHT/MINIAOD/UL2018_MiniAODv2-v1/50000/C6AECDA5-1071-BF4D-A5DC-A172ED56A26D.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018C-UL2018_MiniAODv2-v1/out_1082.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/JetHT/MINIAOD/UL2018_MiniAODv2-v1/50000/8FD1E784-A891-0C47-862A-83CA644626E3.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018C-UL2018_MiniAODv2-v1/out_597.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/JetHT/MINIAOD/UL2018_MiniAODv2-v1/50000/3C9BFE9E-7113-7A43-BCF1-852DE9B8C35B.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018C-UL2018_MiniAODv2-v1/out_596.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/JetHT/MINIAOD/UL2018_MiniAODv2-v1/40000/15F9ED5C-5937-EC42-9814-64A01601DB16.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018C-UL2018_MiniAODv2-v1/out_195.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/JetHT/MINIAOD/UL2018_MiniAODv2-v1/40000/DECA116F-FD7A-7142-B4AB-B3B01553E97F.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018C-UL2018_MiniAODv2-v1/out_194.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/JetHT/MINIAOD/UL2018_MiniAODv2-v1/40000/0228BDDF-EB63-E948-9FD8-7982A11A6AE0.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018C-UL2018_MiniAODv2-v1/out_197.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/JetHT/MINIAOD/UL2018_MiniAODv2-v1/40000/F77D0EBA-5417-D44B-BFF2-6E6FDA76B2A5.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018C-UL2018_MiniAODv2-v1/out_196.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/JetHT/MINIAOD/UL2018_MiniAODv2-v1/40000/B84D3BF3-46ED-E949-96E1-984A9D5658C1.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018C-UL2018_MiniAODv2-v1/out_191.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/JetHT/MINIAOD/UL2018_MiniAODv2-v1/40000/E83A844E-ABA5-AA46-8DF9-F1E8D31494EB.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018C-UL2018_MiniAODv2-v1/out_190.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/JetHT/MINIAOD/UL2018_MiniAODv2-v1/40000/F8F5EBCB-0631-C24D-91BD-6A451A7263DA.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018C-UL2018_MiniAODv2-v1/out_193.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/JetHT/MINIAOD/UL2018_MiniAODv2-v1/40000/8716C65E-A11E-E04A-9CBD-AE28C4B4C0C6.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018C-UL2018_MiniAODv2-v1/out_192.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/JetHT/MINIAOD/UL2018_MiniAODv2-v1/280000/01D5BE16-98D5-D54B-B213-8C13B22E1B47.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018C-UL2018_MiniAODv2-v1/out_270.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/JetHT/MINIAOD/UL2018_MiniAODv2-v1/280000/E8B0D2C6-95D4-0440-BAD3-2CCF6319090E.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018C-UL2018_MiniAODv2-v1/out_271.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/JetHT/MINIAOD/UL2018_MiniAODv2-v1/280000/7F5336C2-E0D7-6F4F-B8C1-57EEA656FE62.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018C-UL2018_MiniAODv2-v1/out_272.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/JetHT/MINIAOD/UL2018_MiniAODv2-v1/280000/45A224D2-C279-9F4A-A353-08700D660112.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018C-UL2018_MiniAODv2-v1/out_273.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/JetHT/MINIAOD/UL2018_MiniAODv2-v1/280000/96EE6F0E-BDBE-CD41-B8E1-9F0F9B4EBE85.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018C-UL2018_MiniAODv2-v1/out_274.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/JetHT/MINIAOD/UL2018_MiniAODv2-v1/280000/1A2F63D1-852D-AE40-A8B7-303AE44B8A3F.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018C-UL2018_MiniAODv2-v1/out_275.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/JetHT/MINIAOD/UL2018_MiniAODv2-v1/280000/628EFC33-241A-CA42-9E7D-18A4FDBF8CBB.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018C-UL2018_MiniAODv2-v1/out_276.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/JetHT/MINIAOD/UL2018_MiniAODv2-v1/280000/42408E5C-E43A-5B4D-8E48-05E88DE456B3.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018C-UL2018_MiniAODv2-v1/out_277.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/JetHT/MINIAOD/UL2018_MiniAODv2-v1/280000/DB198834-AA02-7344-962E-7A2ABC710A22.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018C-UL2018_MiniAODv2-v1/out_278.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/JetHT/MINIAOD/UL2018_MiniAODv2-v1/280000/9E8F1A74-1BC0-1E4A-A68F-923E43B6473E.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018C-UL2018_MiniAODv2-v1/out_279.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/JetHT/MINIAOD/UL2018_MiniAODv2-v1/50000/5AEB9069-FC9D-4640-B3FE-1F905F7F718E.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018C-UL2018_MiniAODv2-v1/out_569.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260000/3C1CA92A-1268-104E-93BB-80004EE06F89.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018C-UL2018_MiniAODv2-v1/out_497.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260000/D8FC0FA1-0832-9E4C-8DE8-6EC6295CD48E.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018C-UL2018_MiniAODv2-v1/out_524.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260000/1CA43D4B-86F3-E14B-A99E-08672B2789D8.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018C-UL2018_MiniAODv2-v1/out_525.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260000/5DE16B52-AD0A-B04E-A1CF-5A484CEEFE9D.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018C-UL2018_MiniAODv2-v1/out_526.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260000/36D058AE-076E-7A4F-9DAC-78CA6B52B128.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018C-UL2018_MiniAODv2-v1/out_527.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260000/E3A728C8-A2D4-F947-9C10-8A55C919FE48.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018C-UL2018_MiniAODv2-v1/out_520.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260000/20FFD227-2C87-9340-AEDD-52989CB553AF.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018C-UL2018_MiniAODv2-v1/out_521.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260000/816798DE-226B-B947-8203-997405FE5B99.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018C-UL2018_MiniAODv2-v1/out_522.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260000/D92CF0C5-8EE9-644D-AB4E-A5C7AF143114.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018C-UL2018_MiniAODv2-v1/out_523.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/JetHT/MINIAOD/UL2018_MiniAODv2-v1/50000/A0B0B727-63AA-994F-A117-5EBF9F1DB624.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018C-UL2018_MiniAODv2-v1/out_1014.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/JetHT/MINIAOD/UL2018_MiniAODv2-v1/50000/E1C67BB7-7670-B442-A3C2-D7B5481F60FC.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018C-UL2018_MiniAODv2-v1/out_1015.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/JetHT/MINIAOD/UL2018_MiniAODv2-v1/50000/FA519DFD-1597-E242-A856-F8C14C18BC80.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018C-UL2018_MiniAODv2-v1/out_599.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/JetHT/MINIAOD/UL2018_MiniAODv2-v1/50000/755502F2-4C8A-374C-AB99-0C7B66A4C7A3.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018C-UL2018_MiniAODv2-v1/out_1017.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260000/4BC91151-6C72-C94C-8101-57C3DA22B350.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018C-UL2018_MiniAODv2-v1/out_528.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260000/29F9CFC3-4C1C-1E4D-ABE4-53BA6292C76F.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018C-UL2018_MiniAODv2-v1/out_529.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/JetHT/MINIAOD/UL2018_MiniAODv2-v1/50000/063B171C-B55D-E74A-B5E2-FC3B6BF4B39D.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018C-UL2018_MiniAODv2-v1/out_1012.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/JetHT/MINIAOD/UL2018_MiniAODv2-v1/50000/B46A2568-9B20-7346-ACEA-19F928FCCF4B.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018C-UL2018_MiniAODv2-v1/out_1013.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/JetHT/MINIAOD/UL2018_MiniAODv2-v1/50000/27657E31-5D57-3A41-A507-8349539EE8CB.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018C-UL2018_MiniAODv2-v1/out_449.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/JetHT/MINIAOD/UL2018_MiniAODv2-v1/50000/3B394234-12BC-9444-8582-6656E36F57DA.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018C-UL2018_MiniAODv2-v1/out_448.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/JetHT/MINIAOD/UL2018_MiniAODv2-v1/50000/EC7943F5-E343-7048-A698-C99F1D222C70.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018C-UL2018_MiniAODv2-v1/out_345.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/JetHT/MINIAOD/UL2018_MiniAODv2-v1/50000/F27182C8-F13D-2D41-BFED-4EE9BB178F9B.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018C-UL2018_MiniAODv2-v1/out_443.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/JetHT/MINIAOD/UL2018_MiniAODv2-v1/50000/90107589-008F-EB48-BC4E-0D7E133E79B0.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018C-UL2018_MiniAODv2-v1/out_442.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/JetHT/MINIAOD/UL2018_MiniAODv2-v1/50000/3B465C09-5936-C241-BD5C-3E9C4DC86CA3.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018C-UL2018_MiniAODv2-v1/out_441.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/JetHT/MINIAOD/UL2018_MiniAODv2-v1/50000/20A37C58-F75E-424F-8AA9-FB415012584C.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018C-UL2018_MiniAODv2-v1/out_440.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/JetHT/MINIAOD/UL2018_MiniAODv2-v1/50000/603F62B2-AEF6-2546-A8FC-D4F20FAA72D7.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018C-UL2018_MiniAODv2-v1/out_447.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/JetHT/MINIAOD/UL2018_MiniAODv2-v1/50000/A3564AEA-AB8F-6F47-84CD-C7AE9D6B8E44.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018C-UL2018_MiniAODv2-v1/out_446.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/JetHT/MINIAOD/UL2018_MiniAODv2-v1/50000/6232DEDB-ED4E-FD4D-82B2-7B5E40E8E48D.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018C-UL2018_MiniAODv2-v1/out_445.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/JetHT/MINIAOD/UL2018_MiniAODv2-v1/50000/55416DB2-C4D7-554D-A7B6-57F7557351F1.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018C-UL2018_MiniAODv2-v1/out_444.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/JetHT/MINIAOD/UL2018_MiniAODv2-v1/50000/6B379F66-086A-5B4B-B4AA-EC8DAE8C467D.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018C-UL2018_MiniAODv2-v1/out_47.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/JetHT/MINIAOD/UL2018_MiniAODv2-v1/50000/CB83F26B-D677-5D45-BCC4-1C1B3DD93B5A.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018C-UL2018_MiniAODv2-v1/out_108.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/JetHT/MINIAOD/UL2018_MiniAODv2-v1/50000/C74A485F-6C02-6748-B3C5-F5D57DF3710F.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018C-UL2018_MiniAODv2-v1/out_109.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/JetHT/MINIAOD/UL2018_MiniAODv2-v1/50000/1A1F010A-0991-7D42-846F-F5EAF8AB615D.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018C-UL2018_MiniAODv2-v1/out_588.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/JetHT/MINIAOD/UL2018_MiniAODv2-v1/50000/C4116857-5B5F-B343-8299-74BD3E7E3021.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018C-UL2018_MiniAODv2-v1/out_102.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/JetHT/MINIAOD/UL2018_MiniAODv2-v1/50000/FE85CDAB-5A1F-DD4F-9D2B-3051EFEF4DFA.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018C-UL2018_MiniAODv2-v1/out_103.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/JetHT/MINIAOD/UL2018_MiniAODv2-v1/50000/89BBC658-5D29-D54E-94B9-5732B2D3D965.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018C-UL2018_MiniAODv2-v1/out_100.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/JetHT/MINIAOD/UL2018_MiniAODv2-v1/50000/EA2B137A-B71C-7245-B8A0-0C197688CDB6.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018C-UL2018_MiniAODv2-v1/out_101.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/JetHT/MINIAOD/UL2018_MiniAODv2-v1/50000/2B576C54-55D6-1C4B-A538-B1AD5054B89C.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018C-UL2018_MiniAODv2-v1/out_106.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/JetHT/MINIAOD/UL2018_MiniAODv2-v1/50000/38B4056F-E74C-D44C-9A32-ECEE2B3C0FD8.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018C-UL2018_MiniAODv2-v1/out_107.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/JetHT/MINIAOD/UL2018_MiniAODv2-v1/50000/6FE05857-6E77-B144-8109-C72C7268F751.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018C-UL2018_MiniAODv2-v1/out_104.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/JetHT/MINIAOD/UL2018_MiniAODv2-v1/50000/42589677-67D2-E948-9D49-CB24C3BA8B03.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018C-UL2018_MiniAODv2-v1/out_105.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/JetHT/MINIAOD/UL2018_MiniAODv2-v1/280000/05AE4EF5-FE75-CC41-B72B-3EB2208B7680.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018C-UL2018_MiniAODv2-v1/out_902.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/JetHT/MINIAOD/UL2018_MiniAODv2-v1/280000/D1FAA4B8-48DB-C94D-9C99-274E2A658EE2.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018C-UL2018_MiniAODv2-v1/out_903.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/JetHT/MINIAOD/UL2018_MiniAODv2-v1/280000/DB3DB403-B661-0646-B490-688FC930DC21.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018C-UL2018_MiniAODv2-v1/out_39.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/JetHT/MINIAOD/UL2018_MiniAODv2-v1/280000/6BE358B8-6D6A-D04B-A92F-994616E1DF44.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018C-UL2018_MiniAODv2-v1/out_38.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/JetHT/MINIAOD/UL2018_MiniAODv2-v1/280000/7F8A793B-94F0-8543-9BAB-F43F7D782CFB.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018C-UL2018_MiniAODv2-v1/out_906.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/JetHT/MINIAOD/UL2018_MiniAODv2-v1/280000/6061B125-64B6-5244-8A89-732B52AF1641.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018C-UL2018_MiniAODv2-v1/out_907.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/JetHT/MINIAOD/UL2018_MiniAODv2-v1/280000/F847C0B2-6B80-AB47-8AC0-0658597AD636.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018C-UL2018_MiniAODv2-v1/out_904.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/JetHT/MINIAOD/UL2018_MiniAODv2-v1/280000/4AD2481B-34A2-6D41-8F62-4B5D5D7BB78F.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018C-UL2018_MiniAODv2-v1/out_905.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/JetHT/MINIAOD/UL2018_MiniAODv2-v1/50000/DEBB7EB7-B90A-0F49-BA3E-33929720FC9A.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018C-UL2018_MiniAODv2-v1/out_33.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/JetHT/MINIAOD/UL2018_MiniAODv2-v1/50000/874551A3-44E5-C945-89D7-1975E6E8E3CE.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018C-UL2018_MiniAODv2-v1/out_32.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/JetHT/MINIAOD/UL2018_MiniAODv2-v1/50000/B9E5E9AF-5C56-4F4B-AC9D-BC6795F4E651.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018C-UL2018_MiniAODv2-v1/out_31.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/JetHT/MINIAOD/UL2018_MiniAODv2-v1/50000/703BFC09-EF8A-F04A-B78C-D4020E8B7086.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018C-UL2018_MiniAODv2-v1/out_30.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/JetHT/MINIAOD/UL2018_MiniAODv2-v1/280000/0E538044-8EAF-6A46-9179-0CAEC8A5CA11.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018C-UL2018_MiniAODv2-v1/out_37.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/JetHT/MINIAOD/UL2018_MiniAODv2-v1/50000/9C8C53E5-C175-564D-9651-557000946348.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018C-UL2018_MiniAODv2-v1/out_36.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/JetHT/MINIAOD/UL2018_MiniAODv2-v1/50000/056412BC-628D-EC4B-870A-74467596BD35.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018C-UL2018_MiniAODv2-v1/out_35.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/JetHT/MINIAOD/UL2018_MiniAODv2-v1/50000/DD690D29-DDFF-D14A-B749-8E872B2710CA.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018C-UL2018_MiniAODv2-v1/out_34.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/JetHT/MINIAOD/UL2018_MiniAODv2-v1/50000/3CFB21AC-8DBC-0943-9ED0-BB6E67C20A7F.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018C-UL2018_MiniAODv2-v1/out_641.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/JetHT/MINIAOD/UL2018_MiniAODv2-v1/50000/634172DF-4288-8948-9483-8AFA2BEDBF3B.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018C-UL2018_MiniAODv2-v1/out_640.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/JetHT/MINIAOD/UL2018_MiniAODv2-v1/50000/61960665-58B4-1F4D-AC61-FED3ED8E56CA.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018C-UL2018_MiniAODv2-v1/out_643.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/JetHT/MINIAOD/UL2018_MiniAODv2-v1/50000/AECFC3DF-EA8E-B249-AAAC-6798B6360F36.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018C-UL2018_MiniAODv2-v1/out_642.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/JetHT/MINIAOD/UL2018_MiniAODv2-v1/50000/BDB5513A-3E99-7D49-AB94-BE242C546176.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018C-UL2018_MiniAODv2-v1/out_645.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/JetHT/MINIAOD/UL2018_MiniAODv2-v1/50000/40DD41CF-84BB-C641-9096-FD84BA62B771.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018C-UL2018_MiniAODv2-v1/out_644.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/JetHT/MINIAOD/UL2018_MiniAODv2-v1/50000/9C0ABAA0-2825-D24E-A0C6-3FEB2FD4BF27.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018C-UL2018_MiniAODv2-v1/out_438.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/JetHT/MINIAOD/UL2018_MiniAODv2-v1/50000/CED169B2-ACEC-7444-81B6-85FE44DDAF53.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018C-UL2018_MiniAODv2-v1/out_439.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/JetHT/MINIAOD/UL2018_MiniAODv2-v1/50000/75BE5061-51AB-AD44-8F47-B4267FB862CC.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018C-UL2018_MiniAODv2-v1/out_436.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/JetHT/MINIAOD/UL2018_MiniAODv2-v1/50000/B4197DCE-AF23-D44B-ABAB-6D0EFA433652.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018C-UL2018_MiniAODv2-v1/out_437.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/JetHT/MINIAOD/UL2018_MiniAODv2-v1/50000/1EBCF0BA-838E-5246-951E-4C756276D24C.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018C-UL2018_MiniAODv2-v1/out_434.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/JetHT/MINIAOD/UL2018_MiniAODv2-v1/50000/59FA9546-1A8F-2E46-8FA9-66631CA621B7.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018C-UL2018_MiniAODv2-v1/out_435.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/JetHT/MINIAOD/UL2018_MiniAODv2-v1/50000/6E97ED9A-C5B3-6D46-8F1D-9FD88FFE4809.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018C-UL2018_MiniAODv2-v1/out_432.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/JetHT/MINIAOD/UL2018_MiniAODv2-v1/50000/8299818B-0679-F845-A613-8457AD56CD1E.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018C-UL2018_MiniAODv2-v1/out_433.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/JetHT/MINIAOD/UL2018_MiniAODv2-v1/50000/01303258-7F18-3340-97A4-0F0FBFC0359B.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018C-UL2018_MiniAODv2-v1/out_430.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/JetHT/MINIAOD/UL2018_MiniAODv2-v1/50000/3A1BA7A6-B2ED-A94B-8EB9-8125ACC0F1B9.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018C-UL2018_MiniAODv2-v1/out_431.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/JetHT/MINIAOD/UL2018_MiniAODv2-v1/40000/0A3B51B6-374A-F045-896D-36ED4CEE8487.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018C-UL2018_MiniAODv2-v1/out_826.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/JetHT/MINIAOD/UL2018_MiniAODv2-v1/40000/03060175-2F93-7E43-B834-E6161CFF8E34.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018C-UL2018_MiniAODv2-v1/out_339.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/JetHT/MINIAOD/UL2018_MiniAODv2-v1/40000/1BD0E4D5-4D9E-FF4A-A50E-0B0A9E1F5AD3.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018C-UL2018_MiniAODv2-v1/out_338.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/JetHT/MINIAOD/UL2018_MiniAODv2-v1/40000/DFEC2640-1450-1B4A-BE5F-1719AE03E52E.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018C-UL2018_MiniAODv2-v1/out_335.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/JetHT/MINIAOD/UL2018_MiniAODv2-v1/40000/089F1618-200B-C348-91B3-9EB797895838.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018C-UL2018_MiniAODv2-v1/out_334.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/JetHT/MINIAOD/UL2018_MiniAODv2-v1/40000/7909400F-73B8-0245-8652-C0227F27F6CE.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018C-UL2018_MiniAODv2-v1/out_337.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/JetHT/MINIAOD/UL2018_MiniAODv2-v1/40000/3DED5CD4-1913-194B-B351-B0343E49F1DA.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018C-UL2018_MiniAODv2-v1/out_336.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/JetHT/MINIAOD/UL2018_MiniAODv2-v1/40000/2B4E30B9-E720-3241-B7E3-CB1DC3944B30.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018C-UL2018_MiniAODv2-v1/out_331.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/JetHT/MINIAOD/UL2018_MiniAODv2-v1/40000/68C7DF1F-BE8E-6349-9DA7-31D0FCDDFEC5.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018C-UL2018_MiniAODv2-v1/out_330.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/JetHT/MINIAOD/UL2018_MiniAODv2-v1/40000/C2CD3609-0C88-5749-92FB-B51E1FFDD72E.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018C-UL2018_MiniAODv2-v1/out_333.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/JetHT/MINIAOD/UL2018_MiniAODv2-v1/40000/744D47F1-7B41-164C-BB04-BD724BA2F081.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018C-UL2018_MiniAODv2-v1/out_332.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/JetHT/MINIAOD/UL2018_MiniAODv2-v1/50000/2F49768C-BDF9-F543-AB6E-0DB29C840C41.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018C-UL2018_MiniAODv2-v1/out_559.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/JetHT/MINIAOD/UL2018_MiniAODv2-v1/50000/BF4D84D3-568D-3849-A6B9-CFB596FA1DEF.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018C-UL2018_MiniAODv2-v1/out_1026.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/JetHT/MINIAOD/UL2018_MiniAODv2-v1/50000/0BFEB71B-C2ED-0A41-B4A4-A61327ED7D90.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018C-UL2018_MiniAODv2-v1/out_745.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/JetHT/MINIAOD/UL2018_MiniAODv2-v1/280000/2BDE86EE-E6B3-A049-9B85-80CCB76FB09E.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018C-UL2018_MiniAODv2-v1/out_854.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/JetHT/MINIAOD/UL2018_MiniAODv2-v1/40000/33C22EE3-C10A-F542-A35E-E9B3B1718962.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018C-UL2018_MiniAODv2-v1/out_818.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/JetHT/MINIAOD/UL2018_MiniAODv2-v1/280000/A62F83E9-7C94-6748-A6C5-C069744819E9.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018C-UL2018_MiniAODv2-v1/out_856.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/JetHT/MINIAOD/UL2018_MiniAODv2-v1/280000/4D1335C3-22AF-B24F-B673-8D17B758DC1D.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018C-UL2018_MiniAODv2-v1/out_857.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/JetHT/MINIAOD/UL2018_MiniAODv2-v1/280000/E0A409E0-2FF0-F745-80D9-1585A9C7C929.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018C-UL2018_MiniAODv2-v1/out_850.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/JetHT/MINIAOD/UL2018_MiniAODv2-v1/280000/4404C41C-796E-474C-8252-D1976C67EC7B.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018C-UL2018_MiniAODv2-v1/out_851.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/JetHT/MINIAOD/UL2018_MiniAODv2-v1/280000/0E688107-641D-1942-8A32-047779F00FE1.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018C-UL2018_MiniAODv2-v1/out_852.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/JetHT/MINIAOD/UL2018_MiniAODv2-v1/50000/A6A19F36-BCEE-6F46-AF67-FE97A7C75BA2.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018C-UL2018_MiniAODv2-v1/out_0.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/JetHT/MINIAOD/UL2018_MiniAODv2-v1/280000/DCB1873E-D869-E744-8965-A5421F0BC396.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018C-UL2018_MiniAODv2-v1/out_858.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/JetHT/MINIAOD/UL2018_MiniAODv2-v1/280000/D3A281DE-7C6C-4246-94BE-45CB960D61BA.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018C-UL2018_MiniAODv2-v1/out_859.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/JetHT/MINIAOD/UL2018_MiniAODv2-v1/50000/6FA5A9FC-DC45-844B-B5F6-35B588C9BE80.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018C-UL2018_MiniAODv2-v1/out_748.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/JetHT/MINIAOD/UL2018_MiniAODv2-v1/50000/809B48EB-20C7-A44A-A7F0-B36361A46A9E.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018C-UL2018_MiniAODv2-v1/out_6.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/JetHT/MINIAOD/UL2018_MiniAODv2-v1/40000/83B8CFF8-85C3-9F47-A6C9-CA4C29924872.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018C-UL2018_MiniAODv2-v1/out_151.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/JetHT/MINIAOD/UL2018_MiniAODv2-v1/50000/589075EF-57D3-5241-A3B6-A22B239BAF29.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018C-UL2018_MiniAODv2-v1/out_425.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/JetHT/MINIAOD/UL2018_MiniAODv2-v1/50000/0EB19872-CF3E-C743-9DE5-C8FF280D8716.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018C-UL2018_MiniAODv2-v1/out_43.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/JetHT/MINIAOD/UL2018_MiniAODv2-v1/50000/C2834786-2F03-254C-8F70-8B328AA047C6.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018C-UL2018_MiniAODv2-v1/out_99.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/JetHT/MINIAOD/UL2018_MiniAODv2-v1/50000/10B8041F-FB63-5744-B7B8-8E3E014E7A02.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018C-UL2018_MiniAODv2-v1/out_98.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/JetHT/MINIAOD/UL2018_MiniAODv2-v1/50000/D0A9C4F4-0259-F64D-AF22-2527F186CCA9.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018C-UL2018_MiniAODv2-v1/out_8.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/JetHT/MINIAOD/UL2018_MiniAODv2-v1/50000/411D19D7-4F86-1E4F-A2EC-CDF7044E27CA.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018C-UL2018_MiniAODv2-v1/out_91.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/JetHT/MINIAOD/UL2018_MiniAODv2-v1/50000/3A8F1AA1-C405-CA49-AFA2-371F58A674B5.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018C-UL2018_MiniAODv2-v1/out_90.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/JetHT/MINIAOD/UL2018_MiniAODv2-v1/50000/B539C744-6BC0-474E-9CFB-85092D1EBF73.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018C-UL2018_MiniAODv2-v1/out_93.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/JetHT/MINIAOD/UL2018_MiniAODv2-v1/50000/A1E2C448-5490-164C-AB49-CE10E09A7242.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018C-UL2018_MiniAODv2-v1/out_92.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/JetHT/MINIAOD/UL2018_MiniAODv2-v1/50000/BCB24948-730C-C744-A5A9-E7F878B592D2.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018C-UL2018_MiniAODv2-v1/out_95.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/JetHT/MINIAOD/UL2018_MiniAODv2-v1/50000/1989D87B-46E8-F545-AC78-785B8C37B8FA.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018C-UL2018_MiniAODv2-v1/out_94.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/JetHT/MINIAOD/UL2018_MiniAODv2-v1/50000/1B1DF413-213B-664C-8102-67FC9CCCE2DE.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018C-UL2018_MiniAODv2-v1/out_97.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/JetHT/MINIAOD/UL2018_MiniAODv2-v1/50000/0EBAEE9F-8242-9E45-B3B9-80EE20165916.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018C-UL2018_MiniAODv2-v1/out_96.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/JetHT/MINIAOD/UL2018_MiniAODv2-v1/40000/88F24BB1-36AA-2243-87CB-DA21F48B697C.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018C-UL2018_MiniAODv2-v1/out_150.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/JetHT/MINIAOD/UL2018_MiniAODv2-v1/50000/22B1528B-2506-D943-BD83-2A44D9F196CA.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018C-UL2018_MiniAODv2-v1/out_424.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/JetHT/MINIAOD/UL2018_MiniAODv2-v1/40000/976C3547-5EF7-504A-BEAB-378C4A4C21B5.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018C-UL2018_MiniAODv2-v1/out_814.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/JetHT/MINIAOD/UL2018_MiniAODv2-v1/50000/5E6885AC-D601-DF47-B375-5F966579C132.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018C-UL2018_MiniAODv2-v1/out_314.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/JetHT/MINIAOD/UL2018_MiniAODv2-v1/40000/26EB39B4-6968-DE41-852B-708AC9308DE1.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018C-UL2018_MiniAODv2-v1/out_815.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/JetHT/MINIAOD/UL2018_MiniAODv2-v1/40000/B33E6A2A-4EC2-C441-BA4D-D606E717BF1E.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018C-UL2018_MiniAODv2-v1/out_153.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/JetHT/MINIAOD/UL2018_MiniAODv2-v1/50000/C0437940-389B-144C-9DC6-3AF44956128F.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018C-UL2018_MiniAODv2-v1/out_740.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/JetHT/MINIAOD/UL2018_MiniAODv2-v1/50000/223B89B0-03CB-B94C-9921-A7E2008B6E58.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018C-UL2018_MiniAODv2-v1/out_741.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/JetHT/MINIAOD/UL2018_MiniAODv2-v1/50000/9144E867-625F-2743-8D47-5F381C0DF10B.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018C-UL2018_MiniAODv2-v1/out_742.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/JetHT/MINIAOD/UL2018_MiniAODv2-v1/50000/1A4E863B-18A9-F14C-8C02-BF3B9368AFFE.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018C-UL2018_MiniAODv2-v1/out_743.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/JetHT/MINIAOD/UL2018_MiniAODv2-v1/50000/6C5B40CC-1B3E-4D4D-A958-53699F3FB389.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018C-UL2018_MiniAODv2-v1/out_744.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/JetHT/MINIAOD/UL2018_MiniAODv2-v1/50000/2305ADD6-FDFD-FD43-8342-F46764829125.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018C-UL2018_MiniAODv2-v1/out_558.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/JetHT/MINIAOD/UL2018_MiniAODv2-v1/50000/1D2394A3-7FAE-264C-8979-A662ADB0B3E4.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018C-UL2018_MiniAODv2-v1/out_746.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/JetHT/MINIAOD/UL2018_MiniAODv2-v1/50000/32D0B9F3-A5D0-F940-A551-6C7146FC09A6.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018C-UL2018_MiniAODv2-v1/out_747.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/JetHT/MINIAOD/UL2018_MiniAODv2-v1/50000/DE1A8EA9-3941-7B43-941E-85037E44961C.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018C-UL2018_MiniAODv2-v1/out_555.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/JetHT/MINIAOD/UL2018_MiniAODv2-v1/50000/A2742BDD-B79C-3242-B39D-B92ABDAD078C.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018C-UL2018_MiniAODv2-v1/out_554.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/JetHT/MINIAOD/UL2018_MiniAODv2-v1/50000/12998E03-DBE7-0D43-9047-EBADDBE34236.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018C-UL2018_MiniAODv2-v1/out_557.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/JetHT/MINIAOD/UL2018_MiniAODv2-v1/50000/E7C2C416-2E71-C74A-AFB2-97CFE70D9AFF.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018C-UL2018_MiniAODv2-v1/out_556.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/JetHT/MINIAOD/UL2018_MiniAODv2-v1/50000/C9D9377D-6837-0842-9D68-A4ADE46E6544.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018C-UL2018_MiniAODv2-v1/out_551.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/JetHT/MINIAOD/UL2018_MiniAODv2-v1/50000/3973DA09-157C-6344-AE5F-40901E4A9928.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018C-UL2018_MiniAODv2-v1/out_550.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/JetHT/MINIAOD/UL2018_MiniAODv2-v1/50000/D7ACE3AD-0A20-7046-87E9-1AE4284060E6.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018C-UL2018_MiniAODv2-v1/out_553.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/JetHT/MINIAOD/UL2018_MiniAODv2-v1/50000/05ABCD4C-4800-9347-A1CB-50BDF3B1CB13.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018C-UL2018_MiniAODv2-v1/out_552.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/JetHT/MINIAOD/UL2018_MiniAODv2-v1/40000/74A8ACC2-453A-114C-9802-44D49AC0F69A.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018C-UL2018_MiniAODv2-v1/out_238.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/JetHT/MINIAOD/UL2018_MiniAODv2-v1/40000/DCAF26AF-6CA2-C84E-8F6F-2A7F273B7597.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018C-UL2018_MiniAODv2-v1/out_239.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/JetHT/MINIAOD/UL2018_MiniAODv2-v1/40000/C292610E-A8BB-8E4C-BEEB-D6DD6810E460.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018C-UL2018_MiniAODv2-v1/out_234.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/JetHT/MINIAOD/UL2018_MiniAODv2-v1/40000/FDBFAAB1-2153-1F45-9334-A35EA6843CAA.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018C-UL2018_MiniAODv2-v1/out_235.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/JetHT/MINIAOD/UL2018_MiniAODv2-v1/40000/54A7E152-3185-CC4E-B9BE-ACBCF9C65529.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018C-UL2018_MiniAODv2-v1/out_236.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/JetHT/MINIAOD/UL2018_MiniAODv2-v1/40000/AF38C9BA-A4A8-234D-92E4-9185009AF837.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018C-UL2018_MiniAODv2-v1/out_237.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/JetHT/MINIAOD/UL2018_MiniAODv2-v1/40000/20A91FA4-589E-6349-87F8-0C2345C30BFD.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018C-UL2018_MiniAODv2-v1/out_230.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/JetHT/MINIAOD/UL2018_MiniAODv2-v1/40000/1601BBF6-BA0A-254F-8A37-7DBDAED3F7AD.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018C-UL2018_MiniAODv2-v1/out_231.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/JetHT/MINIAOD/UL2018_MiniAODv2-v1/40000/5A733BFE-56A9-5747-A9CD-C727BFE8F30B.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018C-UL2018_MiniAODv2-v1/out_232.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/JetHT/MINIAOD/UL2018_MiniAODv2-v1/40000/4DD03084-23F8-EF45-BB33-DD0417B99F2F.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018C-UL2018_MiniAODv2-v1/out_233.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/JetHT/MINIAOD/UL2018_MiniAODv2-v1/50000/600DE94F-206E-744F-A7E6-F2C024212B5D.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018C-UL2018_MiniAODv2-v1/out_1050.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/JetHT/MINIAOD/UL2018_MiniAODv2-v1/50000/5CB5D88B-54A3-EB4A-BF3A-DA4607024E2C.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018C-UL2018_MiniAODv2-v1/out_1051.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/JetHT/MINIAOD/UL2018_MiniAODv2-v1/50000/2CD1E0DA-B299-CC45-ABC1-39E5FDFF4CEF.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018C-UL2018_MiniAODv2-v1/out_1052.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/JetHT/MINIAOD/UL2018_MiniAODv2-v1/50000/52D4BA2F-1751-3C40-BEB3-EA3AAF9ABFE0.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018C-UL2018_MiniAODv2-v1/out_1053.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/JetHT/MINIAOD/UL2018_MiniAODv2-v1/50000/46658054-F1C3-434C-B952-E07D91A2605E.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018C-UL2018_MiniAODv2-v1/out_1054.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/JetHT/MINIAOD/UL2018_MiniAODv2-v1/50000/B8395A1F-B6B3-324D-9942-628D3486ABC6.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018C-UL2018_MiniAODv2-v1/out_1055.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/JetHT/MINIAOD/UL2018_MiniAODv2-v1/50000/F58977D2-1262-194D-8D81-A6022E531718.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018C-UL2018_MiniAODv2-v1/out_1056.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/JetHT/MINIAOD/UL2018_MiniAODv2-v1/50000/DB1CA149-18B2-9842-A189-616F56D0C53E.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018C-UL2018_MiniAODv2-v1/out_1057.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/JetHT/MINIAOD/UL2018_MiniAODv2-v1/50000/505D7C3C-3D56-2643-8A7C-1227A24A3718.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018C-UL2018_MiniAODv2-v1/out_1058.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/JetHT/MINIAOD/UL2018_MiniAODv2-v1/50000/14B5885F-382A-B147-9116-6B8534BB057B.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018C-UL2018_MiniAODv2-v1/out_1059.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/JetHT/MINIAOD/UL2018_MiniAODv2-v1/50000/23F2BE97-24FE-6045-8E2E-C97EF4120AD0.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018C-UL2018_MiniAODv2-v1/out_1.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/JetHT/MINIAOD/UL2018_MiniAODv2-v1/410000/3DD09B31-09A4-6344-9A50-83599F7D935A.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018C-UL2018_MiniAODv2-v1/out_1159.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/JetHT/MINIAOD/UL2018_MiniAODv2-v1/40000/4A7DDE69-F0E2-464F-AE3C-FD99F481BA7B.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018C-UL2018_MiniAODv2-v1/out_155.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/JetHT/MINIAOD/UL2018_MiniAODv2-v1/40000/7D7608CE-31F0-3B42-BED7-7C811C209402.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018C-UL2018_MiniAODv2-v1/out_146.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/JetHT/MINIAOD/UL2018_MiniAODv2-v1/40000/057D2373-4ACE-3D44-AFF2-CAEF06F39DFE.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018C-UL2018_MiniAODv2-v1/out_147.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/JetHT/MINIAOD/UL2018_MiniAODv2-v1/40000/6A5D20C5-3D60-9D43-8FBC-E80B259FC000.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018C-UL2018_MiniAODv2-v1/out_144.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/JetHT/MINIAOD/UL2018_MiniAODv2-v1/40000/6888B8DE-D3CC-6B4F-A41A-5E092F4CF569.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018C-UL2018_MiniAODv2-v1/out_145.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/JetHT/MINIAOD/UL2018_MiniAODv2-v1/40000/F19C9142-F2F0-7749-B30C-A430580813DE.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018C-UL2018_MiniAODv2-v1/out_142.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/JetHT/MINIAOD/UL2018_MiniAODv2-v1/40000/1A50E546-6555-EE43-BDB7-C696B81F3E11.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018C-UL2018_MiniAODv2-v1/out_143.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/JetHT/MINIAOD/UL2018_MiniAODv2-v1/40000/84B404F9-404D-9344-95DB-B82AD0633110.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018C-UL2018_MiniAODv2-v1/out_140.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/JetHT/MINIAOD/UL2018_MiniAODv2-v1/40000/13409FB7-4FFA-374B-B040-6A27AF23CC0E.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018C-UL2018_MiniAODv2-v1/out_141.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/JetHT/MINIAOD/UL2018_MiniAODv2-v1/50000/AF95E208-6202-E94C-807B-67F83CB78D92.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018C-UL2018_MiniAODv2-v1/out_612.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/JetHT/MINIAOD/UL2018_MiniAODv2-v1/50000/2FAF1DB0-AD80-114E-9F15-97AC2DB5E5B3.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018C-UL2018_MiniAODv2-v1/out_613.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/JetHT/MINIAOD/UL2018_MiniAODv2-v1/50000/DC205356-9323-574B-8944-41A3D76B6710.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018C-UL2018_MiniAODv2-v1/out_610.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/JetHT/MINIAOD/UL2018_MiniAODv2-v1/50000/A0C317C4-026E-AC41-BA9D-6A8E3E5202EB.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018C-UL2018_MiniAODv2-v1/out_611.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/JetHT/MINIAOD/UL2018_MiniAODv2-v1/50000/9E4D37B2-6E79-074C-9D3D-F78A5AF04FE0.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018C-UL2018_MiniAODv2-v1/out_616.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/JetHT/MINIAOD/UL2018_MiniAODv2-v1/50000/C21615DF-C9D9-FA42-902B-2F00C232B0D9.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018C-UL2018_MiniAODv2-v1/out_617.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/JetHT/MINIAOD/UL2018_MiniAODv2-v1/40000/6FA50BAD-23D8-D341-A8E2-D68BD957B5BA.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018C-UL2018_MiniAODv2-v1/out_148.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/JetHT/MINIAOD/UL2018_MiniAODv2-v1/40000/E493CBE5-9180-CF4D-845A-0D5F06969416.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018C-UL2018_MiniAODv2-v1/out_149.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260000/A7504C1D-BF92-954F-A494-C6DD604ADF40.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018C-UL2018_MiniAODv2-v1/out_511.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/JetHT/MINIAOD/UL2018_MiniAODv2-v1/50000/77750591-6E2A-D34B-B119-4E71B7932F30.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018C-UL2018_MiniAODv2-v1/out_615.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260000/A25EAB5D-AA97-B147-9437-A76B921EAA69.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018C-UL2018_MiniAODv2-v1/out_510.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/JetHT/MINIAOD/UL2018_MiniAODv2-v1/40000/5A390FEC-66FA-5F46-830D-4BECD97C8498.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018C-UL2018_MiniAODv2-v1/out_154.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260000/EDAD0B0F-B93D-A846-A358-802B2829E617.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018C-UL2018_MiniAODv2-v1/out_513.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260000/E203B229-B025-974F-9AEB-C3479B9168C7.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018C-UL2018_MiniAODv2-v1/out_512.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/JetHT/MINIAOD/UL2018_MiniAODv2-v1/40000/B6407A81-3634-2947-8087-7D3F746F5FBE.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018C-UL2018_MiniAODv2-v1/out_948.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/JetHT/MINIAOD/UL2018_MiniAODv2-v1/40000/25550544-DFA8-1F42-9A1A-4895503D8427.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018C-UL2018_MiniAODv2-v1/out_949.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/JetHT/MINIAOD/UL2018_MiniAODv2-v1/40000/8DC53AFA-8D43-B641-9833-3C98FA861609.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018C-UL2018_MiniAODv2-v1/out_946.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/JetHT/MINIAOD/UL2018_MiniAODv2-v1/40000/70551A41-3D0C-F04C-9B79-9638A154B3B1.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018C-UL2018_MiniAODv2-v1/out_947.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/JetHT/MINIAOD/UL2018_MiniAODv2-v1/40000/9B056D37-B9B3-FA45-AD9B-9122DB9BF957.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018C-UL2018_MiniAODv2-v1/out_944.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/JetHT/MINIAOD/UL2018_MiniAODv2-v1/40000/3C11CBF9-48C8-BE47-9774-DC5FEC9A0F90.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018C-UL2018_MiniAODv2-v1/out_945.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/JetHT/MINIAOD/UL2018_MiniAODv2-v1/40000/8206FD68-8164-6145-A569-4186BFEEE4ED.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018C-UL2018_MiniAODv2-v1/out_942.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/JetHT/MINIAOD/UL2018_MiniAODv2-v1/40000/9AC8DF11-0012-B24A-9522-70F526093F06.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018C-UL2018_MiniAODv2-v1/out_943.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/JetHT/MINIAOD/UL2018_MiniAODv2-v1/40000/83297BF5-AA46-B740-AB83-920E2EE3F4DA.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018C-UL2018_MiniAODv2-v1/out_940.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/JetHT/MINIAOD/UL2018_MiniAODv2-v1/40000/B3C68AEC-13E5-E84A-B73A-ED311116A899.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018C-UL2018_MiniAODv2-v1/out_941.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/JetHT/MINIAOD/UL2018_MiniAODv2-v1/50000/AF9803F4-30E7-254A-B352-5BC15334B55B.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018C-UL2018_MiniAODv2-v1/out_689.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/JetHT/MINIAOD/UL2018_MiniAODv2-v1/50000/3C4C53B9-1C36-F34D-A0F5-235D488F8DFF.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018C-UL2018_MiniAODv2-v1/out_688.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/JetHT/MINIAOD/UL2018_MiniAODv2-v1/40000/378DFE35-FFC0-0A4E-87AA-1539A148C0D7.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018C-UL2018_MiniAODv2-v1/out_157.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/JetHT/MINIAOD/UL2018_MiniAODv2-v1/50000/7D1C2B2F-CB83-F64D-83FB-F14A18C79F94.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018C-UL2018_MiniAODv2-v1/out_685.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/JetHT/MINIAOD/UL2018_MiniAODv2-v1/50000/47358FFD-6407-E344-B7E3-DBB0E06D3CF3.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018C-UL2018_MiniAODv2-v1/out_684.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/JetHT/MINIAOD/UL2018_MiniAODv2-v1/50000/C3CA8F7B-AFDC-7A4E-AC46-11344A2659A2.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018C-UL2018_MiniAODv2-v1/out_687.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/JetHT/MINIAOD/UL2018_MiniAODv2-v1/50000/93F5145B-B3BB-9149-81BA-F3939DEDDA2E.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018C-UL2018_MiniAODv2-v1/out_686.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/JetHT/MINIAOD/UL2018_MiniAODv2-v1/50000/1E1EF57B-B3E8-A54E-8D8B-1A8ABFCDA407.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018C-UL2018_MiniAODv2-v1/out_681.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/JetHT/MINIAOD/UL2018_MiniAODv2-v1/50000/D1443643-49E6-3647-9DFD-417AB6500400.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018C-UL2018_MiniAODv2-v1/out_680.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/JetHT/MINIAOD/UL2018_MiniAODv2-v1/50000/1D768E85-E6FE-8748-ADBA-162A163153F7.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018C-UL2018_MiniAODv2-v1/out_683.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/JetHT/MINIAOD/UL2018_MiniAODv2-v1/50000/3981FBA5-FB7C-274D-A5B8-D7753D3AE013.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018C-UL2018_MiniAODv2-v1/out_682.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260000/3292092F-F284-7D44-9A7C-9E8961230749.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018C-UL2018_MiniAODv2-v1/out_458.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260000/7191462A-B046-4942-AFC9-193C2CCD5908.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018C-UL2018_MiniAODv2-v1/out_459.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/JetHT/MINIAOD/UL2018_MiniAODv2-v1/40000/D0FB64DA-B485-F246-9615-A1425857DC49.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018C-UL2018_MiniAODv2-v1/out_133.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/JetHT/MINIAOD/UL2018_MiniAODv2-v1/40000/67FA50AD-E37D-D840-A35F-A6728A81B827.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018C-UL2018_MiniAODv2-v1/out_132.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/JetHT/MINIAOD/UL2018_MiniAODv2-v1/50000/06FD9109-0008-034C-818A-CCDBAC5AE5F4.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018C-UL2018_MiniAODv2-v1/out_131.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/JetHT/MINIAOD/UL2018_MiniAODv2-v1/50000/47E4C6A1-F1B8-7745-B954-E946CD154ACB.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018C-UL2018_MiniAODv2-v1/out_130.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/JetHT/MINIAOD/UL2018_MiniAODv2-v1/40000/6D0DCD5A-EB16-2844-B0EE-3BC7E7BBCE06.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018C-UL2018_MiniAODv2-v1/out_137.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/JetHT/MINIAOD/UL2018_MiniAODv2-v1/40000/392551D3-601E-2544-97AF-F45AC074E828.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018C-UL2018_MiniAODv2-v1/out_136.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/JetHT/MINIAOD/UL2018_MiniAODv2-v1/40000/9F5B6591-48C2-C84B-A131-16D06BDE3607.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018C-UL2018_MiniAODv2-v1/out_135.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/JetHT/MINIAOD/UL2018_MiniAODv2-v1/40000/8AA33025-EE90-D24D-A562-96FE70C3A762.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018C-UL2018_MiniAODv2-v1/out_134.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260000/BED2C9BF-65F9-654B-8FD7-641D59DB5C30.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018C-UL2018_MiniAODv2-v1/out_494.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260000/B18741C4-74B7-EB47-BB0B-D83A8DAEAF86.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018C-UL2018_MiniAODv2-v1/out_495.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/JetHT/MINIAOD/UL2018_MiniAODv2-v1/40000/0E3C6B82-222B-2F4D-A9BC-7DEE4757D73B.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018C-UL2018_MiniAODv2-v1/out_139.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/JetHT/MINIAOD/UL2018_MiniAODv2-v1/40000/68FEFBBA-CBBB-6D43-8A17-3DEF3ED241DE.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018C-UL2018_MiniAODv2-v1/out_138.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260000/78A1B067-02A5-CE45-BD8F-F8C016E9A917.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018C-UL2018_MiniAODv2-v1/out_490.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260000/A7BA6DD8-88FF-3D4E-9325-214B764F89DD.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018C-UL2018_MiniAODv2-v1/out_491.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260000/76260BB7-6A06-D44A-9E2D-702CB5051A96.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018C-UL2018_MiniAODv2-v1/out_492.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260000/DA15AB6E-9691-0F45-B125-F994FAA957BA.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018C-UL2018_MiniAODv2-v1/out_493.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260000/499101A4-95E5-A442-A8B0-72AADD47AC84.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018C-UL2018_MiniAODv2-v1/out_24.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/JetHT/MINIAOD/UL2018_MiniAODv2-v1/50000/A6D3B7F3-2004-7F45-9761-9F18954F32DC.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018C-UL2018_MiniAODv2-v1/out_25.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/JetHT/MINIAOD/UL2018_MiniAODv2-v1/50000/467613AA-E0EF-9D43-8985-0BAD5F631759.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018C-UL2018_MiniAODv2-v1/out_26.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/JetHT/MINIAOD/UL2018_MiniAODv2-v1/50000/5934C2EA-4A07-8745-84EE-04D55AC986DD.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018C-UL2018_MiniAODv2-v1/out_27.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260000/71537D12-2D7F-F44C-993F-37B7D003ABE7.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018C-UL2018_MiniAODv2-v1/out_20.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260000/30CB72F7-9283-D547-93F0-FE07030FF052.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018C-UL2018_MiniAODv2-v1/out_21.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260000/ECC0BADF-7E6B-FF4F-B28D-20DCEAB394C2.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018C-UL2018_MiniAODv2-v1/out_22.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260000/271BEE62-BB49-9848-B0CF-5E4FE6EC88C3.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018C-UL2018_MiniAODv2-v1/out_23.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/JetHT/MINIAOD/UL2018_MiniAODv2-v1/50000/84B8EDBF-8949-2347-8E5D-7D51A2663B13.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018C-UL2018_MiniAODv2-v1/out_1029.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/JetHT/MINIAOD/UL2018_MiniAODv2-v1/50000/D2ED1B71-4A6B-A841-9C6B-FC7AA11A39B3.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018C-UL2018_MiniAODv2-v1/out_28.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/JetHT/MINIAOD/UL2018_MiniAODv2-v1/50000/4DAB95C1-E8F6-1F4E-A629-6923DE6F92FD.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018C-UL2018_MiniAODv2-v1/out_29.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/JetHT/MINIAOD/UL2018_MiniAODv2-v1/40000/1459E6B5-46DA-BD43-A349-6D13F4BC5348.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018C-UL2018_MiniAODv2-v1/out_820.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/JetHT/MINIAOD/UL2018_MiniAODv2-v1/50000/CA818671-AD91-BE44-B9F8-40B7C6E95CBF.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018C-UL2018_MiniAODv2-v1/out_407.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/JetHT/MINIAOD/UL2018_MiniAODv2-v1/50000/5B82740E-A9C8-B84B-8D0E-814EE19501AF.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018C-UL2018_MiniAODv2-v1/out_406.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/JetHT/MINIAOD/UL2018_MiniAODv2-v1/50000/CE50FAC9-64C5-5242-B850-571E209294CA.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018C-UL2018_MiniAODv2-v1/out_405.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/JetHT/MINIAOD/UL2018_MiniAODv2-v1/50000/11DED494-AA3E-FF48-A649-5BFDAF47450B.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018C-UL2018_MiniAODv2-v1/out_404.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/JetHT/MINIAOD/UL2018_MiniAODv2-v1/50000/70715225-DE04-EA41-B603-C59B5F9FBD07.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018C-UL2018_MiniAODv2-v1/out_403.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/JetHT/MINIAOD/UL2018_MiniAODv2-v1/50000/D0BCCA70-3A54-6A4B-9664-34DAABAD5C41.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018C-UL2018_MiniAODv2-v1/out_402.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/JetHT/MINIAOD/UL2018_MiniAODv2-v1/50000/741DF00B-1E50-BB4B-BBD5-DCFEA39E6B23.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018C-UL2018_MiniAODv2-v1/out_401.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/JetHT/MINIAOD/UL2018_MiniAODv2-v1/50000/25B4869E-4051-C942-84FF-EF8FDD192338.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018C-UL2018_MiniAODv2-v1/out_400.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/JetHT/MINIAOD/UL2018_MiniAODv2-v1/40000/05C7137B-FD5E-D44F-86A4-DE84AD5E63F0.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018C-UL2018_MiniAODv2-v1/out_933.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/JetHT/MINIAOD/UL2018_MiniAODv2-v1/40000/714581E3-A60F-2E40-ADA3-F08F414AAE07.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018C-UL2018_MiniAODv2-v1/out_932.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/JetHT/MINIAOD/UL2018_MiniAODv2-v1/40000/78CB323F-7E5E-D045-A150-3BB03F4D5023.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018C-UL2018_MiniAODv2-v1/out_931.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/JetHT/MINIAOD/UL2018_MiniAODv2-v1/40000/CEF6CFA8-922F-804B-A876-583B68FCB258.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018C-UL2018_MiniAODv2-v1/out_930.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/JetHT/MINIAOD/UL2018_MiniAODv2-v1/40000/7D6248E6-EEAE-114D-AF62-AD33FA46EA61.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018C-UL2018_MiniAODv2-v1/out_937.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/JetHT/MINIAOD/UL2018_MiniAODv2-v1/50000/6F01096C-FBC9-9643-9BC7-DFC415C45156.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018C-UL2018_MiniAODv2-v1/out_452.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/JetHT/MINIAOD/UL2018_MiniAODv2-v1/50000/33665EA0-3470-814F-93CC-BA512B16307A.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018C-UL2018_MiniAODv2-v1/out_409.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/JetHT/MINIAOD/UL2018_MiniAODv2-v1/50000/A0B00DCB-41C3-6A44-93DC-B8D57426C7D2.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018C-UL2018_MiniAODv2-v1/out_408.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/JetHT/MINIAOD/UL2018_MiniAODv2-v1/50000/146B77C4-A849-6E42-A9BC-FC57284FCB49.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018C-UL2018_MiniAODv2-v1/out_453.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/JetHT/MINIAOD/UL2018_MiniAODv2-v1/50000/3F3A5D61-B736-FA45-BBC8-AAC060F814C3.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018C-UL2018_MiniAODv2-v1/out_414.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/JetHT/MINIAOD/UL2018_MiniAODv2-v1/50000/B92904AF-0840-3845-B8CE-D28DC0029470.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018C-UL2018_MiniAODv2-v1/out_455.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/JetHT/MINIAOD/UL2018_MiniAODv2-v1/410000/84905DC7-DE3D-884C-BE71-B218A04E140D.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018C-UL2018_MiniAODv2-v1/out_1118.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/JetHT/MINIAOD/UL2018_MiniAODv2-v1/40000/62708D51-94B1-D34E-9A0A-F9127E9BCE37.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018C-UL2018_MiniAODv2-v1/out_183.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/JetHT/MINIAOD/UL2018_MiniAODv2-v1/280000/C9F7B437-1C30-4D4E-A2A5-B1679FF911F5.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018C-UL2018_MiniAODv2-v1/out_849.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/JetHT/MINIAOD/UL2018_MiniAODv2-v1/50000/669A428C-FC74-6749-8C51-628C86422439.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018C-UL2018_MiniAODv2-v1/out_415.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/JetHT/MINIAOD/UL2018_MiniAODv2-v1/50000/37A9ACCE-7AD8-4E42-B2C7-7B02AA165C60.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018C-UL2018_MiniAODv2-v1/out_737.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/JetHT/MINIAOD/UL2018_MiniAODv2-v1/50000/18F2F660-08F5-B84A-A77F-771B7D6C0509.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018C-UL2018_MiniAODv2-v1/out_379.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/JetHT/MINIAOD/UL2018_MiniAODv2-v1/50000/9EA449D1-79EC-A64A-AD9C-922BA0D98A3C.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018C-UL2018_MiniAODv2-v1/out_378.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/JetHT/MINIAOD/UL2018_MiniAODv2-v1/40000/85F6890D-D607-1947-9C75-1690BD337296.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018C-UL2018_MiniAODv2-v1/out_228.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/JetHT/MINIAOD/UL2018_MiniAODv2-v1/40000/479A9123-8C95-FB40-901F-2AE2CB873773.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018C-UL2018_MiniAODv2-v1/out_829.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/JetHT/MINIAOD/UL2018_MiniAODv2-v1/40000/A4617F52-C7CE-C24E-953F-AB5FB8A2E8A2.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018C-UL2018_MiniAODv2-v1/out_828.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/JetHT/MINIAOD/UL2018_MiniAODv2-v1/50000/DE53F094-DD28-B642-BBCE-1C27AE037292.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018C-UL2018_MiniAODv2-v1/out_371.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/JetHT/MINIAOD/UL2018_MiniAODv2-v1/50000/6C01F924-A09F-1344-A3B4-C2BAE3C736B3.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018C-UL2018_MiniAODv2-v1/out_370.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/JetHT/MINIAOD/UL2018_MiniAODv2-v1/50000/4751683F-83AF-F44C-AB05-1D6B90811827.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018C-UL2018_MiniAODv2-v1/out_373.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/JetHT/MINIAOD/UL2018_MiniAODv2-v1/50000/AECBFA5E-F1BE-DB49-9E2F-C35B9E688A0C.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018C-UL2018_MiniAODv2-v1/out_372.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/JetHT/MINIAOD/UL2018_MiniAODv2-v1/50000/3B533715-0C32-1F41-9EB7-97DFB6F8A680.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018C-UL2018_MiniAODv2-v1/out_375.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/JetHT/MINIAOD/UL2018_MiniAODv2-v1/50000/26A8433C-E1B6-FE45-B21A-A0903E721F1D.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018C-UL2018_MiniAODv2-v1/out_374.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/JetHT/MINIAOD/UL2018_MiniAODv2-v1/50000/80EADC00-E5A9-4F46-8EC0-D188E7B8992B.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018C-UL2018_MiniAODv2-v1/out_377.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/JetHT/MINIAOD/UL2018_MiniAODv2-v1/50000/F58E4DFA-DD97-F042-9F6B-A4ACFE948983.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018C-UL2018_MiniAODv2-v1/out_376.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/JetHT/MINIAOD/UL2018_MiniAODv2-v1/50000/84CCE2B6-D8A3-B546-AB6C-8253001F35BC.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018C-UL2018_MiniAODv2-v1/out_1177.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/JetHT/MINIAOD/UL2018_MiniAODv2-v1/50000/671D2BDC-2759-AC44-8CBC-12DFBA282A91.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018C-UL2018_MiniAODv2-v1/out_708.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/JetHT/MINIAOD/UL2018_MiniAODv2-v1/50000/7CB4998B-86D2-8A46-A7CD-27CEE7DD66A4.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018C-UL2018_MiniAODv2-v1/out_709.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/JetHT/MINIAOD/UL2018_MiniAODv2-v1/50000/3ADDD4B5-9CE3-B548-A76C-1D3BFC3D87AC.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018C-UL2018_MiniAODv2-v1/out_1176.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/JetHT/MINIAOD/UL2018_MiniAODv2-v1/50000/0E7E4D23-4E12-F747-A900-C37C5C65B273.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018C-UL2018_MiniAODv2-v1/out_704.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/JetHT/MINIAOD/UL2018_MiniAODv2-v1/50000/EF8E45C5-A76C-D04D-9C93-D06816909681.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018C-UL2018_MiniAODv2-v1/out_705.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/JetHT/MINIAOD/UL2018_MiniAODv2-v1/50000/580AF3A0-B6C5-374E-A714-39230101A9A7.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018C-UL2018_MiniAODv2-v1/out_706.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/JetHT/MINIAOD/UL2018_MiniAODv2-v1/50000/56B62D05-DA8C-5D44-B7B8-84353EF429CE.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018C-UL2018_MiniAODv2-v1/out_707.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/JetHT/MINIAOD/UL2018_MiniAODv2-v1/50000/C340BEFD-9E5F-4040-A781-1A6B8F12DE39.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018C-UL2018_MiniAODv2-v1/out_700.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/JetHT/MINIAOD/UL2018_MiniAODv2-v1/40000/BD0B0824-1308-C041-B1D9-5A0CDE800429.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018C-UL2018_MiniAODv2-v1/out_1175.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/JetHT/MINIAOD/UL2018_MiniAODv2-v1/50000/2E26C83C-EDA2-3A43-9D8C-6F7DE5C0CAB1.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018C-UL2018_MiniAODv2-v1/out_702.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/JetHT/MINIAOD/UL2018_MiniAODv2-v1/50000/97100483-EDD5-F144-A9EC-644398DF07D1.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018C-UL2018_MiniAODv2-v1/out_703.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/JetHT/MINIAOD/UL2018_MiniAODv2-v1/50000/7D249EAC-9DDF-1E43-8D0F-1CE995B859A8.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018C-UL2018_MiniAODv2-v1/out_393.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/JetHT/MINIAOD/UL2018_MiniAODv2-v1/50000/BABF085D-5912-9444-84AE-896D78CDF465.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018C-UL2018_MiniAODv2-v1/out_392.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/JetHT/MINIAOD/UL2018_MiniAODv2-v1/50000/BFD4800D-2DD5-8C47-AE67-4936F8558F79.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018C-UL2018_MiniAODv2-v1/out_88.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/JetHT/MINIAOD/UL2018_MiniAODv2-v1/50000/0B28269A-37BB-1D4A-B131-AC50ADA0AE16.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018C-UL2018_MiniAODv2-v1/out_619.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/JetHT/MINIAOD/UL2018_MiniAODv2-v1/50000/D5501DA0-4E89-B548-AB35-7CA48CE2F09F.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018C-UL2018_MiniAODv2-v1/out_397.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/JetHT/MINIAOD/UL2018_MiniAODv2-v1/50000/DA71851C-E721-704C-8350-AE56C1CDE572.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018C-UL2018_MiniAODv2-v1/out_396.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/JetHT/MINIAOD/UL2018_MiniAODv2-v1/50000/B64EA6B0-BBC6-DA4A-A603-B29B8772A805.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018C-UL2018_MiniAODv2-v1/out_395.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/JetHT/MINIAOD/UL2018_MiniAODv2-v1/50000/3F6EC544-91F0-8B4F-A8F5-3ADF3E079357.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018C-UL2018_MiniAODv2-v1/out_394.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/JetHT/MINIAOD/UL2018_MiniAODv2-v1/50000/C7A44813-BB1B-E041-9307-04EEAC035976.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018C-UL2018_MiniAODv2-v1/out_82.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/JetHT/MINIAOD/UL2018_MiniAODv2-v1/50000/07BCB676-4432-324E-86E4-D53522A84D53.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018C-UL2018_MiniAODv2-v1/out_83.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/JetHT/MINIAOD/UL2018_MiniAODv2-v1/50000/16FF62A5-826F-E94A-8878-2E0049A32482.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018C-UL2018_MiniAODv2-v1/out_80.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/JetHT/MINIAOD/UL2018_MiniAODv2-v1/50000/CCA38EE0-A19D-B141-AA8F-60C7D38A4A14.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018C-UL2018_MiniAODv2-v1/out_81.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/JetHT/MINIAOD/UL2018_MiniAODv2-v1/50000/244A273A-2142-0242-B5C2-5E3BBA0D6C1E.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018C-UL2018_MiniAODv2-v1/out_86.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/JetHT/MINIAOD/UL2018_MiniAODv2-v1/50000/BEC51311-3E33-FE48-B166-8495124E1271.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018C-UL2018_MiniAODv2-v1/out_87.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/JetHT/MINIAOD/UL2018_MiniAODv2-v1/50000/C504D69B-DC1B-4A47-9099-F9808CB9AD7A.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018C-UL2018_MiniAODv2-v1/out_84.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/JetHT/MINIAOD/UL2018_MiniAODv2-v1/50000/5E3A74AA-98F8-C147-A941-8EB200F76E46.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018C-UL2018_MiniAODv2-v1/out_85.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/JetHT/MINIAOD/UL2018_MiniAODv2-v1/40000/96EA5BE9-62B7-FD42-89B0-459F1A772D04.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018C-UL2018_MiniAODv2-v1/out_797.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/JetHT/MINIAOD/UL2018_MiniAODv2-v1/40000/0CA42344-8EE3-4D41-9A04-81E5B81B40D8.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018C-UL2018_MiniAODv2-v1/out_796.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/JetHT/MINIAOD/UL2018_MiniAODv2-v1/40000/F1902E9D-0401-034A-B8F6-0126A580512D.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018C-UL2018_MiniAODv2-v1/out_795.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/JetHT/MINIAOD/UL2018_MiniAODv2-v1/40000/C2C268D9-6000-6644-8292-B1C4577B5055.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018C-UL2018_MiniAODv2-v1/out_794.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/JetHT/MINIAOD/UL2018_MiniAODv2-v1/40000/E490BBAC-C7E9-3744-8A51-309524029283.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018C-UL2018_MiniAODv2-v1/out_793.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/JetHT/MINIAOD/UL2018_MiniAODv2-v1/40000/328A9097-0F22-764C-8B04-7D9C77A4337A.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018C-UL2018_MiniAODv2-v1/out_792.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/JetHT/MINIAOD/UL2018_MiniAODv2-v1/40000/CB0A95C4-75CF-D549-8EE5-314C09F6DB53.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018C-UL2018_MiniAODv2-v1/out_791.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/JetHT/MINIAOD/UL2018_MiniAODv2-v1/50000/1BD30316-E3DC-014B-B357-2033E027C943.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018C-UL2018_MiniAODv2-v1/out_790.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/JetHT/MINIAOD/UL2018_MiniAODv2-v1/410000/90DF00B9-2191-814D-80D3-CB6C5D3B5568.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018C-UL2018_MiniAODv2-v1/out_1170.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/JetHT/MINIAOD/UL2018_MiniAODv2-v1/50000/82D4D5A6-6F55-0B46-9124-3DB03A2675CF.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018C-UL2018_MiniAODv2-v1/out_749.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/JetHT/MINIAOD/UL2018_MiniAODv2-v1/40000/3552B906-72E5-4448-B175-594D9E33C2EA.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018C-UL2018_MiniAODv2-v1/out_799.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/JetHT/MINIAOD/UL2018_MiniAODv2-v1/40000/7B067420-DAFB-864E-B107-0FCE3D5140B7.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018C-UL2018_MiniAODv2-v1/out_798.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/JetHT/MINIAOD/UL2018_MiniAODv2-v1/50000/9F525F13-128C-2A4F-A8F5-39B614E1390B.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018C-UL2018_MiniAODv2-v1/out_7.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/JetHT/MINIAOD/UL2018_MiniAODv2-v1/40000/B28259E1-E2C9-4D4D-9E03-8572B9629E90.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018C-UL2018_MiniAODv2-v1/out_170.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/JetHT/MINIAOD/UL2018_MiniAODv2-v1/50000/355BEB08-16FA-2B47-BB65-0B4CC83EAC50.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018C-UL2018_MiniAODv2-v1/out_586.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/JetHT/MINIAOD/UL2018_MiniAODv2-v1/50000/1F901DD5-20B2-4743-A7AA-8860AF33F21C.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018C-UL2018_MiniAODv2-v1/out_587.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/JetHT/MINIAOD/UL2018_MiniAODv2-v1/40000/80330431-C65A-0B44-A2D9-79D2A4C05E7D.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018C-UL2018_MiniAODv2-v1/out_1139.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/JetHT/MINIAOD/UL2018_MiniAODv2-v1/50000/D3D8B1E8-39CF-E74F-97DF-0F08B4A8356F.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018C-UL2018_MiniAODv2-v1/out_585.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/JetHT/MINIAOD/UL2018_MiniAODv2-v1/50000/C30E58EE-591B-2849-98D2-711A3E685652.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018C-UL2018_MiniAODv2-v1/out_582.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/JetHT/MINIAOD/UL2018_MiniAODv2-v1/50000/83E9C5C3-49E0-FC4D-80AA-AA8B83D40683.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018C-UL2018_MiniAODv2-v1/out_583.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/JetHT/MINIAOD/UL2018_MiniAODv2-v1/50000/5A52DD6B-539A-FC40-B03B-F939799EEE01.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018C-UL2018_MiniAODv2-v1/out_580.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/JetHT/MINIAOD/UL2018_MiniAODv2-v1/50000/456A6801-8B90-1F46-8C6E-CD664C9C017B.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018C-UL2018_MiniAODv2-v1/out_581.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/JetHT/MINIAOD/UL2018_MiniAODv2-v1/40000/6A7FB43F-14F8-DF42-9849-AE771248576B.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018C-UL2018_MiniAODv2-v1/out_1133.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/JetHT/MINIAOD/UL2018_MiniAODv2-v1/40000/3D7ED0D9-A09E-1548-8CF6-42FF3FC4D3D4.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018C-UL2018_MiniAODv2-v1/out_1132.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/JetHT/MINIAOD/UL2018_MiniAODv2-v1/410000/232BEB93-4C61-1B4E-BD72-CCCAF217FD23.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018C-UL2018_MiniAODv2-v1/out_1131.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/JetHT/MINIAOD/UL2018_MiniAODv2-v1/410000/366F4D38-F8E1-FD44-A6CC-D399E3BE9459.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018C-UL2018_MiniAODv2-v1/out_1130.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/JetHT/MINIAOD/UL2018_MiniAODv2-v1/410000/7A467986-CD1F-0540-AB2D-0BB77243D709.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018C-UL2018_MiniAODv2-v1/out_1137.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/JetHT/MINIAOD/UL2018_MiniAODv2-v1/410000/3A37FEAF-51F1-C444-8568-17157D2B1843.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018C-UL2018_MiniAODv2-v1/out_1136.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/JetHT/MINIAOD/UL2018_MiniAODv2-v1/410000/D908A69E-9280-8243-9AEB-F90363E12514.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018C-UL2018_MiniAODv2-v1/out_1135.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/JetHT/MINIAOD/UL2018_MiniAODv2-v1/50000/C93AA400-06EF-3446-B512-29C91AF00B6A.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018C-UL2018_MiniAODv2-v1/out_589.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/JetHT/MINIAOD/UL2018_MiniAODv2-v1/40000/871A8585-02B9-4A43-BCFD-75C58257B985.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018C-UL2018_MiniAODv2-v1/out_245.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/JetHT/MINIAOD/UL2018_MiniAODv2-v1/40000/BF047DBF-6F02-5A42-A34D-EAC769170278.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018C-UL2018_MiniAODv2-v1/out_244.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/JetHT/MINIAOD/UL2018_MiniAODv2-v1/40000/ED22B661-6FC9-D24B-8918-BF86CE295B97.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018C-UL2018_MiniAODv2-v1/out_247.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/JetHT/MINIAOD/UL2018_MiniAODv2-v1/40000/7DA61058-F54C-DC4E-93A3-23D9E58925C1.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018C-UL2018_MiniAODv2-v1/out_246.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/JetHT/MINIAOD/UL2018_MiniAODv2-v1/40000/837122B0-D3CE-7947-B78B-F7F85D217AC0.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018C-UL2018_MiniAODv2-v1/out_241.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/JetHT/MINIAOD/UL2018_MiniAODv2-v1/50000/CA2C1251-6241-E241-BA69-B7C943BC76E7.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018C-UL2018_MiniAODv2-v1/out_614.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/JetHT/MINIAOD/UL2018_MiniAODv2-v1/40000/DF290E92-CDFD-A047-A2C0-6F326A242B9A.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018C-UL2018_MiniAODv2-v1/out_243.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/JetHT/MINIAOD/UL2018_MiniAODv2-v1/40000/58FA4458-418B-2B4C-91F9-D61B3E575A6A.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018C-UL2018_MiniAODv2-v1/out_242.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/JetHT/MINIAOD/UL2018_MiniAODv2-v1/50000/93719D6F-A9FC-E644-A2BF-0D685E61A951.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018C-UL2018_MiniAODv2-v1/out_1178.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/JetHT/MINIAOD/UL2018_MiniAODv2-v1/40000/091AE83D-3F13-1941-9CD3-23E1B77F4069.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018C-UL2018_MiniAODv2-v1/out_249.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/JetHT/MINIAOD/UL2018_MiniAODv2-v1/40000/110A858C-4B50-6E45-B66B-816DBE102FC3.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018C-UL2018_MiniAODv2-v1/out_248.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/JetHT/MINIAOD/UL2018_MiniAODv2-v1/50000/B3276556-F295-264B-81DC-1175CA83453E.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018C-UL2018_MiniAODv2-v1/out_972.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/JetHT/MINIAOD/UL2018_MiniAODv2-v1/50000/8148A84B-F1E2-E449-978C-753CD136E3C0.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018C-UL2018_MiniAODv2-v1/out_418.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/JetHT/MINIAOD/UL2018_MiniAODv2-v1/40000/C87CD256-2B44-B844-B4F9-23ADB518E6C7.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018C-UL2018_MiniAODv2-v1/out_224.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/JetHT/MINIAOD/UL2018_MiniAODv2-v1/50000/E8D57E60-82C9-7544-BA5E-6A6BB446B443.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018C-UL2018_MiniAODv2-v1/out_639.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/JetHT/MINIAOD/UL2018_MiniAODv2-v1/50000/FB0B5CFE-77A1-F848-83EB-3D2D69996874.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018C-UL2018_MiniAODv2-v1/out_419.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260000/E149094E-8F46-A846-BA51-3E068767E350.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018C-UL2018_MiniAODv2-v1/out_519.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260000/F2C7C41C-02E6-9745-8410-C30C99461CEA.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018C-UL2018_MiniAODv2-v1/out_518.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/JetHT/MINIAOD/UL2018_MiniAODv2-v1/50000/C00D350D-8621-1F4D-992B-AC48B4327A7C.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018C-UL2018_MiniAODv2-v1/out_1009.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/JetHT/MINIAOD/UL2018_MiniAODv2-v1/50000/1B786E08-5899-7A4C-8995-A86976F7CADB.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018C-UL2018_MiniAODv2-v1/out_1008.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/JetHT/MINIAOD/UL2018_MiniAODv2-v1/50000/22CA1BFB-9AA3-944F-8DA1-0BB18D93083A.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018C-UL2018_MiniAODv2-v1/out_1007.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/JetHT/MINIAOD/UL2018_MiniAODv2-v1/50000/A1AD4718-F333-1F47-9DB3-6534F59D5B34.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018C-UL2018_MiniAODv2-v1/out_1006.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/JetHT/MINIAOD/UL2018_MiniAODv2-v1/50000/DC79DDD6-DD88-A743-9403-F9C79142BEC1.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018C-UL2018_MiniAODv2-v1/out_1005.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/JetHT/MINIAOD/UL2018_MiniAODv2-v1/50000/99E06DF2-C83E-F84B-B895-5E69F2D6835D.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018C-UL2018_MiniAODv2-v1/out_1004.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260000/3A2B5811-0EEA-2C42-9A4C-9AC4DBD34BDB.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018C-UL2018_MiniAODv2-v1/out_515.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/JetHT/MINIAOD/UL2018_MiniAODv2-v1/50000/F8F465B0-FBAA-7B4D-9C92-9E3C67E17538.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018C-UL2018_MiniAODv2-v1/out_1002.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260000/BFBD362D-A253-344A-8937-77B9298ECA9B.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018C-UL2018_MiniAODv2-v1/out_517.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/JetHT/MINIAOD/UL2018_MiniAODv2-v1/50000/E753AE0F-E41E-7748-9C5C-3FF350954049.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018C-UL2018_MiniAODv2-v1/out_1000.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/JetHT/MINIAOD/UL2018_MiniAODv2-v1/50000/211BE6D1-2F91-1E43-911C-B0C068A77FD3.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018C-UL2018_MiniAODv2-v1/out_623.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/JetHT/MINIAOD/UL2018_MiniAODv2-v1/50000/CA8ED86C-CD85-AE4B-B6F4-1B6D0B4F5B51.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018C-UL2018_MiniAODv2-v1/out_622.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/JetHT/MINIAOD/UL2018_MiniAODv2-v1/50000/0A5685B5-4354-2449-8A9E-CE3B583359EF.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018C-UL2018_MiniAODv2-v1/out_621.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260000/42A9B5C9-EBE4-3541-A1DC-C15EFE7A28CD.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018C-UL2018_MiniAODv2-v1/out_514.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/JetHT/MINIAOD/UL2018_MiniAODv2-v1/50000/EAC99287-565C-094F-8115-38DEC8981AAB.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018C-UL2018_MiniAODv2-v1/out_627.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/JetHT/MINIAOD/UL2018_MiniAODv2-v1/50000/A738692C-9652-9945-9855-84B4A7329791.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018C-UL2018_MiniAODv2-v1/out_626.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/JetHT/MINIAOD/UL2018_MiniAODv2-v1/50000/007D6B46-0EA8-5C4D-A0A2-89AA176F169A.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018C-UL2018_MiniAODv2-v1/out_625.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/JetHT/MINIAOD/UL2018_MiniAODv2-v1/50000/B26B47B0-D4BF-F64F-8DDB-9AB270D23446.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018C-UL2018_MiniAODv2-v1/out_624.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/JetHT/MINIAOD/UL2018_MiniAODv2-v1/50000/E2E7E898-2A78-954A-9200-08393FB06BC9.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018C-UL2018_MiniAODv2-v1/out_450.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/JetHT/MINIAOD/UL2018_MiniAODv2-v1/50000/8B3722B4-1D26-8A46-9B10-82D784915B28.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018C-UL2018_MiniAODv2-v1/out_451.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/JetHT/MINIAOD/UL2018_MiniAODv2-v1/50000/3F203325-BF56-6242-94D7-5744EF6BA541.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018C-UL2018_MiniAODv2-v1/out_629.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/JetHT/MINIAOD/UL2018_MiniAODv2-v1/50000/71F5501E-F8F3-0B40-8E18-A7230A9B585B.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018C-UL2018_MiniAODv2-v1/out_628.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/JetHT/MINIAOD/UL2018_MiniAODv2-v1/50000/0D13325A-3E6A-F741-B621-D2CD17E1FF20.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018C-UL2018_MiniAODv2-v1/out_454.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/JetHT/MINIAOD/UL2018_MiniAODv2-v1/50000/5DB043BB-ABC2-0F47-A10F-62674ADFA2FA.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018C-UL2018_MiniAODv2-v1/out_1085.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/JetHT/MINIAOD/UL2018_MiniAODv2-v1/50000/CACD8824-7A4E-CB4B-BB91-7B879E362D54.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018C-UL2018_MiniAODv2-v1/out_456.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260000/5086A8A3-C6AE-F841-B9B6-91083B0C3042.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018C-UL2018_MiniAODv2-v1/out_457.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/JetHT/MINIAOD/UL2018_MiniAODv2-v1/50000/DEEE033D-3FA5-EE43-A3BE-9A45428C2FDA.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018C-UL2018_MiniAODv2-v1/out_1084.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/JetHT/MINIAOD/UL2018_MiniAODv2-v1/40000/C23D2920-D82F-CF4B-A6C7-CEEB5A7BEF23.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018C-UL2018_MiniAODv2-v1/out_179.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/JetHT/MINIAOD/UL2018_MiniAODv2-v1/40000/A577F6E7-5F9D-744C-BC99-19212C7BBAA0.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018C-UL2018_MiniAODv2-v1/out_178.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/JetHT/MINIAOD/UL2018_MiniAODv2-v1/40000/D03619C8-3A74-E241-9339-22C731BE59F8.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018C-UL2018_MiniAODv2-v1/out_177.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/JetHT/MINIAOD/UL2018_MiniAODv2-v1/40000/266050E8-25A2-4745-8DB5-067CE4395BD1.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018C-UL2018_MiniAODv2-v1/out_199.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/JetHT/MINIAOD/UL2018_MiniAODv2-v1/40000/D14858CF-31B2-674F-8EC6-212444E6F0FF.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018C-UL2018_MiniAODv2-v1/out_175.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/JetHT/MINIAOD/UL2018_MiniAODv2-v1/40000/9AAE75B9-0448-B144-9723-17DC953B621C.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018C-UL2018_MiniAODv2-v1/out_174.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/JetHT/MINIAOD/UL2018_MiniAODv2-v1/40000/F7887F56-7CBC-9F4D-ADFB-DBC6872CC0A7.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018C-UL2018_MiniAODv2-v1/out_173.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/JetHT/MINIAOD/UL2018_MiniAODv2-v1/40000/52B33C51-C5BC-4846-B5F7-D6B4D5654A3D.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018C-UL2018_MiniAODv2-v1/out_172.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/JetHT/MINIAOD/UL2018_MiniAODv2-v1/40000/CE3F1ED2-2211-9447-90B8-6714F615868C.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018C-UL2018_MiniAODv2-v1/out_171.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/JetHT/MINIAOD/UL2018_MiniAODv2-v1/50000/6812E938-923F-D648-8ABF-CB5737DA77F8.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018C-UL2018_MiniAODv2-v1/out_594.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/JetHT/MINIAOD/UL2018_MiniAODv2-v1/50000/AD031D08-7680-0B41-83E1-0DD64CA2F57C.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018C-UL2018_MiniAODv2-v1/out_977.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/JetHT/MINIAOD/UL2018_MiniAODv2-v1/40000/490585E5-E16B-A446-AD3E-D8DA201AC22E.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018C-UL2018_MiniAODv2-v1/out_182.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/JetHT/MINIAOD/UL2018_MiniAODv2-v1/50000/C84D8A6A-C904-7A44-A138-30275199C90A.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018C-UL2018_MiniAODv2-v1/out_975.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/JetHT/MINIAOD/UL2018_MiniAODv2-v1/50000/871B54C9-51BF-8A41-82BD-764640CBB231.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018C-UL2018_MiniAODv2-v1/out_974.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/JetHT/MINIAOD/UL2018_MiniAODv2-v1/50000/F12DDE8B-4FBF-4049-9AF3-7C420E4B3010.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018C-UL2018_MiniAODv2-v1/out_973.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/JetHT/MINIAOD/UL2018_MiniAODv2-v1/50000/D83EA305-0A7E-5646-9331-99A44BF9CCBE.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018C-UL2018_MiniAODv2-v1/out_1081.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/JetHT/MINIAOD/UL2018_MiniAODv2-v1/50000/137A5891-79E1-6747-B55A-B1C9E21F5055.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018C-UL2018_MiniAODv2-v1/out_971.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/JetHT/MINIAOD/UL2018_MiniAODv2-v1/50000/9D3CFF47-C756-DD47-9A99-6ECC1930F1CF.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018C-UL2018_MiniAODv2-v1/out_970.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/JetHT/MINIAOD/UL2018_MiniAODv2-v1/50000/710032CD-9D3F-4342-8030-F9566F6861DC.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018C-UL2018_MiniAODv2-v1/out_1080.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/JetHT/MINIAOD/UL2018_MiniAODv2-v1/40000/B3A58535-F9EA-C547-91F7-21C4D1548D14.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018C-UL2018_MiniAODv2-v1/out_180.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/JetHT/MINIAOD/UL2018_MiniAODv2-v1/50000/61883056-627D-EE4A-AF9F-8BA3EC82E800.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018C-UL2018_MiniAODv2-v1/out_979.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/JetHT/MINIAOD/UL2018_MiniAODv2-v1/50000/9940A8C2-79A0-3E4E-8851-07FE4E919D5F.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018C-UL2018_MiniAODv2-v1/out_978.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/JetHT/MINIAOD/UL2018_MiniAODv2-v1/50000/63F7BBDE-3411-624B-9EB6-0FCA0041F5D5.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018C-UL2018_MiniAODv2-v1/out_656.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/JetHT/MINIAOD/UL2018_MiniAODv2-v1/410000/E2E544D7-5EC1-2945-917C-1E649E5EFBE1.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018C-UL2018_MiniAODv2-v1/out_1120.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/JetHT/MINIAOD/UL2018_MiniAODv2-v1/50000/E2B4F313-36C8-3D49-83EF-F39692CED34C.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018C-UL2018_MiniAODv2-v1/out_654.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/JetHT/MINIAOD/UL2018_MiniAODv2-v1/40000/8ABB735F-711D-8C43-B148-E75B834522B1.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018C-UL2018_MiniAODv2-v1/out_181.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/JetHT/MINIAOD/UL2018_MiniAODv2-v1/40000/208E6CC4-BD85-9C42-A184-4B4BD14845DC.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018C-UL2018_MiniAODv2-v1/out_186.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/JetHT/MINIAOD/UL2018_MiniAODv2-v1/40000/18B23BF0-7CD7-F24E-87BA-A4BD2262F472.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018C-UL2018_MiniAODv2-v1/out_187.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/JetHT/MINIAOD/UL2018_MiniAODv2-v1/40000/35098B3F-75CC-624E-BF42-5BE255022202.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018C-UL2018_MiniAODv2-v1/out_184.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/JetHT/MINIAOD/UL2018_MiniAODv2-v1/40000/5298AD3E-95BE-5146-9911-40B94216EEA8.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018C-UL2018_MiniAODv2-v1/out_185.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/JetHT/MINIAOD/UL2018_MiniAODv2-v1/50000/7D332059-0505-9A44-A48C-2C18CFC2925C.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018C-UL2018_MiniAODv2-v1/out_652.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/JetHT/MINIAOD/UL2018_MiniAODv2-v1/40000/454E3359-9C26-E84E-9981-42D5407F47B1.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018C-UL2018_MiniAODv2-v1/out_188.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/JetHT/MINIAOD/UL2018_MiniAODv2-v1/40000/7ABDB494-5AD4-AC44-A751-E54E34068DCA.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018C-UL2018_MiniAODv2-v1/out_189.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/JetHT/MINIAOD/UL2018_MiniAODv2-v1/410000/FD20B988-3457-BA41-94EC-F67E924A9FA6.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018C-UL2018_MiniAODv2-v1/out_1122.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/JetHT/MINIAOD/UL2018_MiniAODv2-v1/50000/0DD9F55A-89AF-D841-936D-801B7119F296.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018C-UL2018_MiniAODv2-v1/out_658.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260000/0BD17367-28EB-A940-B8AC-7841CEDFEE55.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018C-UL2018_MiniAODv2-v1/out_516.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/JetHT/MINIAOD/UL2018_MiniAODv2-v1/410000/E4802AE6-C03A-C242-A4B6-43211A9E2BB6.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018C-UL2018_MiniAODv2-v1/out_1123.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/JetHT/MINIAOD/UL2018_MiniAODv2-v1/50000/CE66FB6B-7C09-6D4E-8AAB-15A538E1FD29.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018C-UL2018_MiniAODv2-v1/out_650.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/JetHT/MINIAOD/UL2018_MiniAODv2-v1/410000/3AB15E38-FD17-6D44-B29C-3894947D923E.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018C-UL2018_MiniAODv2-v1/out_1124.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/JetHT/MINIAOD/UL2018_MiniAODv2-v1/50000/F1AEB4CE-D7B5-124C-AFFE-29F9268A36D5.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018C-UL2018_MiniAODv2-v1/out_651.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/JetHT/MINIAOD/UL2018_MiniAODv2-v1/410000/6F817468-6231-E74E-883E-382E925E0D85.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018C-UL2018_MiniAODv2-v1/out_1125.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/JetHT/MINIAOD/UL2018_MiniAODv2-v1/410000/17C76EC0-557B-034E-BD7B-F9E97B378477.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018C-UL2018_MiniAODv2-v1/out_1089.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/JetHT/MINIAOD/UL2018_MiniAODv2-v1/410000/BFC2E670-B5F1-624E-A11F-0DC13D10C54D.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018C-UL2018_MiniAODv2-v1/out_1088.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/JetHT/MINIAOD/UL2018_MiniAODv2-v1/50000/E4B1F305-992D-BE4A-9836-12B6E14CD94B.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018C-UL2018_MiniAODv2-v1/out_11.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/JetHT/MINIAOD/UL2018_MiniAODv2-v1/50000/26B6000F-3688-7443-B100-88D095C34169.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018C-UL2018_MiniAODv2-v1/out_10.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/JetHT/MINIAOD/UL2018_MiniAODv2-v1/50000/65C2754C-9706-3D49-B2EA-0C2CC61A9CFE.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018C-UL2018_MiniAODv2-v1/out_13.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/JetHT/MINIAOD/UL2018_MiniAODv2-v1/50000/2F6201C6-E2B1-E144-A599-5E594E44EB5E.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018C-UL2018_MiniAODv2-v1/out_12.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/JetHT/MINIAOD/UL2018_MiniAODv2-v1/50000/F1885A48-0A50-FF4B-BE7C-ED27339468B4.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018C-UL2018_MiniAODv2-v1/out_15.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/JetHT/MINIAOD/UL2018_MiniAODv2-v1/50000/08433414-6102-B24F-9156-D2D6136352F3.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018C-UL2018_MiniAODv2-v1/out_14.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/JetHT/MINIAOD/UL2018_MiniAODv2-v1/50000/C98597EB-666A-9542-BD3C-938005147159.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018C-UL2018_MiniAODv2-v1/out_17.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/JetHT/MINIAOD/UL2018_MiniAODv2-v1/50000/1BB9EC35-207E-9E4F-AE8D-CA00425DBC9E.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018C-UL2018_MiniAODv2-v1/out_16.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/JetHT/MINIAOD/UL2018_MiniAODv2-v1/50000/333392DE-2CB0-FB41-8371-EC3C6B99F77F.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018C-UL2018_MiniAODv2-v1/out_19.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/JetHT/MINIAOD/UL2018_MiniAODv2-v1/50000/FD2920BC-433C-3B4A-BC41-488CE34681EC.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018C-UL2018_MiniAODv2-v1/out_18.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/JetHT/MINIAOD/UL2018_MiniAODv2-v1/280000/5EFE4637-9985-AA48-85CA-8B585BD758DD.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018C-UL2018_MiniAODv2-v1/out_863.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/JetHT/MINIAOD/UL2018_MiniAODv2-v1/280000/A422971D-EA50-5E40-BFD7-AC17FB5AE8C4.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018C-UL2018_MiniAODv2-v1/out_862.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/JetHT/MINIAOD/UL2018_MiniAODv2-v1/280000/7C72DBB7-0558-824B-88CF-70325FF90EBA.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018C-UL2018_MiniAODv2-v1/out_865.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/JetHT/MINIAOD/UL2018_MiniAODv2-v1/280000/BD47956C-82BD-6448-BDF4-FD234BD70394.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018C-UL2018_MiniAODv2-v1/out_864.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/JetHT/MINIAOD/UL2018_MiniAODv2-v1/280000/1F6F5F5D-DE81-5C45-B2FA-3E41ED2CCCC5.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018C-UL2018_MiniAODv2-v1/out_867.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/JetHT/MINIAOD/UL2018_MiniAODv2-v1/280000/00AAA492-48D3-D84A-B0E8-6ED68BDFADA4.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018C-UL2018_MiniAODv2-v1/out_866.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/JetHT/MINIAOD/UL2018_MiniAODv2-v1/50000/BBFAD70A-6784-A44B-BD2D-02C5916E9B77.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018C-UL2018_MiniAODv2-v1/out_354.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/JetHT/MINIAOD/UL2018_MiniAODv2-v1/50000/2DF6A3A6-6634-8E4B-9C9E-AEF2DA9EDE78.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018C-UL2018_MiniAODv2-v1/out_659.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/JetHT/MINIAOD/UL2018_MiniAODv2-v1/50000/04C1F941-6BB7-7047-A3FD-6D683D847872.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018C-UL2018_MiniAODv2-v1/out_44.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/JetHT/MINIAOD/UL2018_MiniAODv2-v1/280000/C33BE3F3-4D2B-8544-A2E0-A4FE0957BFD9.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018C-UL2018_MiniAODv2-v1/out_883.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/JetHT/MINIAOD/UL2018_MiniAODv2-v1/280000/746EF6E2-4099-9445-85E2-9B3894DB398B.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018C-UL2018_MiniAODv2-v1/out_882.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/JetHT/MINIAOD/UL2018_MiniAODv2-v1/280000/3B68051A-20AC-2948-AFD3-8331531A5D8F.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018C-UL2018_MiniAODv2-v1/out_881.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/JetHT/MINIAOD/UL2018_MiniAODv2-v1/280000/40653872-4899-7149-B639-A32F4C4DA1CB.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018C-UL2018_MiniAODv2-v1/out_880.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/JetHT/MINIAOD/UL2018_MiniAODv2-v1/280000/DA44DFC4-73E7-B943-BF68-4AAFAAE9B65B.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018C-UL2018_MiniAODv2-v1/out_887.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/JetHT/MINIAOD/UL2018_MiniAODv2-v1/280000/A648817B-D450-7B42-B18F-D70B7F72C6B4.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018C-UL2018_MiniAODv2-v1/out_886.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/JetHT/MINIAOD/UL2018_MiniAODv2-v1/280000/09320766-4F18-E547-A3EA-5F07ACC7ADE2.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018C-UL2018_MiniAODv2-v1/out_885.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/JetHT/MINIAOD/UL2018_MiniAODv2-v1/280000/22F7AE90-DB45-AB4B-9E4C-E4FC27D49F22.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018C-UL2018_MiniAODv2-v1/out_884.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/JetHT/MINIAOD/UL2018_MiniAODv2-v1/280000/358DDBCD-5201-6743-BB2E-E72E23EAAB38.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018C-UL2018_MiniAODv2-v1/out_889.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/JetHT/MINIAOD/UL2018_MiniAODv2-v1/280000/378042ED-4E8E-F54F-9681-15B90BA2F1A3.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018C-UL2018_MiniAODv2-v1/out_888.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/JetHT/MINIAOD/UL2018_MiniAODv2-v1/50000/E0D3133A-8B86-544D-8850-4CF6B9259ACF.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018C-UL2018_MiniAODv2-v1/out_116.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/JetHT/MINIAOD/UL2018_MiniAODv2-v1/50000/B9D1F947-15D9-A048-9137-1988D009529D.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018C-UL2018_MiniAODv2-v1/out_45.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/JetHT/MINIAOD/UL2018_MiniAODv2-v1/50000/BCCF8DF4-D066-3E44-8944-ACFF0B6800EB.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018C-UL2018_MiniAODv2-v1/out_657.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/JetHT/MINIAOD/UL2018_MiniAODv2-v1/50000/998CC503-642E-1D4A-B064-847A649CD3F2.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018C-UL2018_MiniAODv2-v1/out_1083.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/JetHT/MINIAOD/UL2018_MiniAODv2-v1/50000/5BC4D873-0FD6-134B-A3C5-0AAF686EED70.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018C-UL2018_MiniAODv2-v1/out_355.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/JetHT/MINIAOD/UL2018_MiniAODv2-v1/40000/7225A08A-EDBD-AA45-8C4D-617802CDCE04.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018C-UL2018_MiniAODv2-v1/out_322.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/JetHT/MINIAOD/UL2018_MiniAODv2-v1/40000/58FED2E7-D125-5541-90E3-793A13041F7B.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018C-UL2018_MiniAODv2-v1/out_323.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/JetHT/MINIAOD/UL2018_MiniAODv2-v1/40000/40A721E7-E951-C544-96B6-0134774A27B0.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018C-UL2018_MiniAODv2-v1/out_320.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/JetHT/MINIAOD/UL2018_MiniAODv2-v1/40000/52CB8CA7-0FD8-1744-8ACE-2FD84A9D7499.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018C-UL2018_MiniAODv2-v1/out_321.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/JetHT/MINIAOD/UL2018_MiniAODv2-v1/40000/C4A3BA34-7D33-0940-8153-E4F08DE700A2.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018C-UL2018_MiniAODv2-v1/out_326.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/JetHT/MINIAOD/UL2018_MiniAODv2-v1/40000/B6FC84EA-622B-D248-8833-84C2A42F9C0E.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018C-UL2018_MiniAODv2-v1/out_327.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/JetHT/MINIAOD/UL2018_MiniAODv2-v1/40000/0E309014-E388-ED4B-A70F-A0925BD94895.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018C-UL2018_MiniAODv2-v1/out_324.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/JetHT/MINIAOD/UL2018_MiniAODv2-v1/40000/0A080788-1C6C-774B-8035-0C847CE066B2.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018C-UL2018_MiniAODv2-v1/out_325.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/JetHT/MINIAOD/UL2018_MiniAODv2-v1/40000/EE8A1435-F69C-C048-A8F8-ED8F33B65C5E.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018C-UL2018_MiniAODv2-v1/out_328.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/JetHT/MINIAOD/UL2018_MiniAODv2-v1/40000/3D33D1F7-F853-6045-919C-54CB3011F563.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018C-UL2018_MiniAODv2-v1/out_329.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/JetHT/MINIAOD/UL2018_MiniAODv2-v1/50000/569961B4-880F-1C43-A8AD-15DD67CF8813.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018C-UL2018_MiniAODv2-v1/out_562.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/JetHT/MINIAOD/UL2018_MiniAODv2-v1/40000/5F70C1BD-B191-6D4E-907E-4303C938AD8E.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018C-UL2018_MiniAODv2-v1/out_201.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/JetHT/MINIAOD/UL2018_MiniAODv2-v1/40000/651FC32C-CA75-164C-8B2C-8C20BD830248.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018C-UL2018_MiniAODv2-v1/out_200.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/JetHT/MINIAOD/UL2018_MiniAODv2-v1/40000/E4235845-BE4B-BD46-ACBD-E60454B7C387.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018C-UL2018_MiniAODv2-v1/out_203.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/JetHT/MINIAOD/UL2018_MiniAODv2-v1/40000/90761D39-04F6-C84C-BCB2-953D433A0F1A.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018C-UL2018_MiniAODv2-v1/out_202.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/JetHT/MINIAOD/UL2018_MiniAODv2-v1/40000/128735A9-C2D0-3749-BB1A-D4797C56C4D1.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018C-UL2018_MiniAODv2-v1/out_205.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/JetHT/MINIAOD/UL2018_MiniAODv2-v1/40000/1AA5807A-AA1A-7449-A11B-7E2191F21D93.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018C-UL2018_MiniAODv2-v1/out_204.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/JetHT/MINIAOD/UL2018_MiniAODv2-v1/40000/934B8954-C354-2B41-9C9A-0D2661780312.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018C-UL2018_MiniAODv2-v1/out_207.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/JetHT/MINIAOD/UL2018_MiniAODv2-v1/40000/E0C157D4-03DF-5144-8934-001115A5609A.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018C-UL2018_MiniAODv2-v1/out_206.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/JetHT/MINIAOD/UL2018_MiniAODv2-v1/40000/8A1F7E43-B196-B442-951B-532AA49CEBBE.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018C-UL2018_MiniAODv2-v1/out_209.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/JetHT/MINIAOD/UL2018_MiniAODv2-v1/40000/4FF5BCAF-4A55-8849-9C67-74CA7D606E18.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018C-UL2018_MiniAODv2-v1/out_208.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/JetHT/MINIAOD/UL2018_MiniAODv2-v1/50000/30980D01-EC97-0041-8ABC-E7A23E72CD70.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018C-UL2018_MiniAODv2-v1/out_779.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/JetHT/MINIAOD/UL2018_MiniAODv2-v1/50000/3964BFB0-275D-3845-8014-2377295322E2.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018C-UL2018_MiniAODv2-v1/out_778.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/JetHT/MINIAOD/UL2018_MiniAODv2-v1/50000/5B33C414-D67F-2D44-B123-5E8116EF4CEB.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018C-UL2018_MiniAODv2-v1/out_679.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/JetHT/MINIAOD/UL2018_MiniAODv2-v1/50000/1FFA9973-D6E0-694A-A2F4-E8BC4967B107.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018C-UL2018_MiniAODv2-v1/out_77.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/JetHT/MINIAOD/UL2018_MiniAODv2-v1/50000/EF76D869-8E0B-204D-BACE-D0DD5F51FDB0.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018C-UL2018_MiniAODv2-v1/out_76.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/JetHT/MINIAOD/UL2018_MiniAODv2-v1/50000/80C9652F-F852-B242-8466-962E6C5CF06C.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018C-UL2018_MiniAODv2-v1/out_75.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/JetHT/MINIAOD/UL2018_MiniAODv2-v1/50000/45D3C71E-158A-5145-BD18-5846089579EF.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018C-UL2018_MiniAODv2-v1/out_74.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/JetHT/MINIAOD/UL2018_MiniAODv2-v1/50000/292BF735-F20E-D846-BEB7-CFCA826A1D7B.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018C-UL2018_MiniAODv2-v1/out_73.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/JetHT/MINIAOD/UL2018_MiniAODv2-v1/50000/FE7BCC8E-AB50-E04F-A85A-01D06DA01355.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018C-UL2018_MiniAODv2-v1/out_72.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/JetHT/MINIAOD/UL2018_MiniAODv2-v1/50000/43AD79BC-C443-FA4B-914F-EA15F4D561C0.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018C-UL2018_MiniAODv2-v1/out_71.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/JetHT/MINIAOD/UL2018_MiniAODv2-v1/50000/53B3F794-B9C5-574C-B7BE-699D546A38DA.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018C-UL2018_MiniAODv2-v1/out_70.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/JetHT/MINIAOD/UL2018_MiniAODv2-v1/50000/BDFA4C4C-52BC-F946-B283-D62A79C38C0B.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018C-UL2018_MiniAODv2-v1/out_655.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/JetHT/MINIAOD/UL2018_MiniAODv2-v1/50000/C24E8513-A834-4D49-92D3-F18E07BC9702.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018C-UL2018_MiniAODv2-v1/out_567.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/JetHT/MINIAOD/UL2018_MiniAODv2-v1/50000/FCB969CA-27BC-E74B-9D73-41D949F2C9DE.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018C-UL2018_MiniAODv2-v1/out_79.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/JetHT/MINIAOD/UL2018_MiniAODv2-v1/50000/EEFDA34E-ACF9-C644-86D2-172868E2DDD5.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018C-UL2018_MiniAODv2-v1/out_78.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/JetHT/MINIAOD/UL2018_MiniAODv2-v1/50000/163B0041-915F-B54D-BA1A-762AD7871403.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018C-UL2018_MiniAODv2-v1/out_1043.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/JetHT/MINIAOD/UL2018_MiniAODv2-v1/50000/4584440C-E8A6-C845-AC86-BB1221689022.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018C-UL2018_MiniAODv2-v1/out_1042.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/JetHT/MINIAOD/UL2018_MiniAODv2-v1/50000/16C75BF6-83BB-7649-AAE5-AC50975226E1.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018C-UL2018_MiniAODv2-v1/out_1041.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/JetHT/MINIAOD/UL2018_MiniAODv2-v1/50000/B19A3E30-755A-C240-81E9-F7F4C87A1867.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018C-UL2018_MiniAODv2-v1/out_1040.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/JetHT/MINIAOD/UL2018_MiniAODv2-v1/50000/A8A0825A-564C-2741-947C-5FE0893A5718.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018C-UL2018_MiniAODv2-v1/out_1047.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/JetHT/MINIAOD/UL2018_MiniAODv2-v1/50000/4A7D9FC2-94DF-CA4E-894D-465223F98B1D.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018C-UL2018_MiniAODv2-v1/out_1046.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/JetHT/MINIAOD/UL2018_MiniAODv2-v1/50000/D2E1269A-BD7B-0C4D-AA3A-A8219468E051.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018C-UL2018_MiniAODv2-v1/out_1045.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/JetHT/MINIAOD/UL2018_MiniAODv2-v1/50000/C9E9F480-3FA3-2341-B61C-E6252D478F64.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018C-UL2018_MiniAODv2-v1/out_1044.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/JetHT/MINIAOD/UL2018_MiniAODv2-v1/50000/B154A7C3-140F-3F49-8DD4-5444DC4F329B.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018C-UL2018_MiniAODv2-v1/out_1049.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/JetHT/MINIAOD/UL2018_MiniAODv2-v1/50000/71728186-1317-6446-8DCE-89B51A2DD3B1.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018C-UL2018_MiniAODv2-v1/out_1048.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/JetHT/MINIAOD/UL2018_MiniAODv2-v1/50000/5C321209-F2A6-8643-85A1-D486E694B700.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018C-UL2018_MiniAODv2-v1/out_359.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/JetHT/MINIAOD/UL2018_MiniAODv2-v1/50000/A98CA1DC-EC11-E840-B77A-CC3BA23BA69D.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018C-UL2018_MiniAODv2-v1/out_358.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/JetHT/MINIAOD/UL2018_MiniAODv2-v1/280000/7D66D2B7-934B-4D4C-8257-D5A2C0E51D7F.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018C-UL2018_MiniAODv2-v1/out_868.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/JetHT/MINIAOD/UL2018_MiniAODv2-v1/410000/032B033A-777F-4D43-8CB8-E612B5B221FD.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018C-UL2018_MiniAODv2-v1/out_1142.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/JetHT/MINIAOD/UL2018_MiniAODv2-v1/410000/0E013E18-6C6A-0145-BD62-AC2DB18A42F7.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018C-UL2018_MiniAODv2-v1/out_1143.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/JetHT/MINIAOD/UL2018_MiniAODv2-v1/410000/068E195A-0F31-8741-A898-0B59B4E37460.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018C-UL2018_MiniAODv2-v1/out_1140.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/JetHT/MINIAOD/UL2018_MiniAODv2-v1/410000/4040FC23-7BFF-EE4E-AD3A-0037F0B14DEB.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018C-UL2018_MiniAODv2-v1/out_1141.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/JetHT/MINIAOD/UL2018_MiniAODv2-v1/410000/AA8A0DAE-70B3-2248-AC9D-8E069FC80074.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018C-UL2018_MiniAODv2-v1/out_1146.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/JetHT/MINIAOD/UL2018_MiniAODv2-v1/410000/C4346E05-7445-2149-9FF1-343CF587EC8B.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018C-UL2018_MiniAODv2-v1/out_1147.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/JetHT/MINIAOD/UL2018_MiniAODv2-v1/410000/3782CA5C-C930-A34D-9163-75DF07CE9129.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018C-UL2018_MiniAODv2-v1/out_1144.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/JetHT/MINIAOD/UL2018_MiniAODv2-v1/40000/773EB942-1390-5348-AF77-3394E0A6AB7C.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018C-UL2018_MiniAODv2-v1/out_1145.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/JetHT/MINIAOD/UL2018_MiniAODv2-v1/50000/A28BA31B-9E98-7040-BC64-6B1C8A7BF78F.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018C-UL2018_MiniAODv2-v1/out_667.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/JetHT/MINIAOD/UL2018_MiniAODv2-v1/50000/85CE7027-91D4-274D-910E-B88DBF8BAC7F.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018C-UL2018_MiniAODv2-v1/out_666.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/JetHT/MINIAOD/UL2018_MiniAODv2-v1/410000/3153FD2A-C6B5-124F-B3C1-B5472D5CDB9F.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018C-UL2018_MiniAODv2-v1/out_1148.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/JetHT/MINIAOD/UL2018_MiniAODv2-v1/410000/912262FA-5F4F-CA4D-989C-2A0B6319245C.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018C-UL2018_MiniAODv2-v1/out_1149.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/JetHT/MINIAOD/UL2018_MiniAODv2-v1/50000/247805BC-E1AA-5145-96F2-0C1EB644C81A.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018C-UL2018_MiniAODv2-v1/out_663.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/JetHT/MINIAOD/UL2018_MiniAODv2-v1/50000/F7313DA2-0C24-7848-963D-1003C2791867.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018C-UL2018_MiniAODv2-v1/out_662.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/JetHT/MINIAOD/UL2018_MiniAODv2-v1/50000/9EE994C6-ACDD-2348-8571-7F02B5350439.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018C-UL2018_MiniAODv2-v1/out_661.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/JetHT/MINIAOD/UL2018_MiniAODv2-v1/50000/71AA8A4C-6CD2-B647-A7F9-A21624B30187.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018C-UL2018_MiniAODv2-v1/out_660.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/JetHT/MINIAOD/UL2018_MiniAODv2-v1/40000/CF917A60-1798-3D40-AEA0-7BECDDA54917.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018C-UL2018_MiniAODv2-v1/out_215.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/JetHT/MINIAOD/UL2018_MiniAODv2-v1/50000/00F86B8C-4316-0C4C-9314-2C3F05039152.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018C-UL2018_MiniAODv2-v1/out_653.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/JetHT/MINIAOD/UL2018_MiniAODv2-v1/50000/C3EF3D4D-5D8B-D740-9813-A070DD9E640A.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018C-UL2018_MiniAODv2-v1/out_1037.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/JetHT/MINIAOD/UL2018_MiniAODv2-v1/50000/9482E930-053D-4C4D-A275-9BC449F1FA5A.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018C-UL2018_MiniAODv2-v1/out_692.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/JetHT/MINIAOD/UL2018_MiniAODv2-v1/50000/49003670-D029-D04E-881C-B53EFB9BE9C7.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018C-UL2018_MiniAODv2-v1/out_693.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/JetHT/MINIAOD/UL2018_MiniAODv2-v1/50000/F0B36FDF-6EEC-B843-A8AB-12016187DDDB.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018C-UL2018_MiniAODv2-v1/out_690.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/JetHT/MINIAOD/UL2018_MiniAODv2-v1/50000/689FA8D1-33EC-B640-966D-444ECF4B2021.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018C-UL2018_MiniAODv2-v1/out_691.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/JetHT/MINIAOD/UL2018_MiniAODv2-v1/50000/5735EF2B-BF85-AF40-85A1-C8D11DA2A5D0.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018C-UL2018_MiniAODv2-v1/out_696.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/JetHT/MINIAOD/UL2018_MiniAODv2-v1/50000/3C2A2532-4572-E045-AB16-13A44D818FBE.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018C-UL2018_MiniAODv2-v1/out_697.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/JetHT/MINIAOD/UL2018_MiniAODv2-v1/50000/0884EA85-6658-2047-AF76-ACBB9D53D1FF.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018C-UL2018_MiniAODv2-v1/out_694.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/JetHT/MINIAOD/UL2018_MiniAODv2-v1/50000/52C4CB92-22FE-D949-AD67-F6F00DE9C837.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018C-UL2018_MiniAODv2-v1/out_695.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/JetHT/MINIAOD/UL2018_MiniAODv2-v1/50000/2818F377-690D-7547-84E6-153B8FFAB4D1.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018C-UL2018_MiniAODv2-v1/out_698.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/JetHT/MINIAOD/UL2018_MiniAODv2-v1/50000/BE630D5F-9D74-F548-860A-AC59D32729A4.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018C-UL2018_MiniAODv2-v1/out_699.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/JetHT/MINIAOD/UL2018_MiniAODv2-v1/40000/6D80943B-F5F3-4243-AEE7-F3DFB4C886B1.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018C-UL2018_MiniAODv2-v1/out_222.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/JetHT/MINIAOD/UL2018_MiniAODv2-v1/50000/5AA2CA82-49A4-074F-B023-9D70A763B61F.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018C-UL2018_MiniAODv2-v1/out_542.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/JetHT/MINIAOD/UL2018_MiniAODv2-v1/50000/E738E2F4-00BC-D242-A751-F3E704BE2FC2.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018C-UL2018_MiniAODv2-v1/out_543.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/JetHT/MINIAOD/UL2018_MiniAODv2-v1/50000/D6D000A4-2F5E-0541-AD49-BE5949384A12.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018C-UL2018_MiniAODv2-v1/out_540.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/JetHT/MINIAOD/UL2018_MiniAODv2-v1/50000/8072F77C-80B5-D748-917B-7D72BDD089C3.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018C-UL2018_MiniAODv2-v1/out_541.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/JetHT/MINIAOD/UL2018_MiniAODv2-v1/50000/57803F87-785D-0847-B22B-B56B6845E5A3.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018C-UL2018_MiniAODv2-v1/out_546.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/JetHT/MINIAOD/UL2018_MiniAODv2-v1/50000/86CC4E10-2EE3-E545-A591-211224C839DD.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018C-UL2018_MiniAODv2-v1/out_547.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/JetHT/MINIAOD/UL2018_MiniAODv2-v1/50000/EFFE96ED-90D8-F141-8BDB-CC50EFFD9F4F.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018C-UL2018_MiniAODv2-v1/out_544.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/JetHT/MINIAOD/UL2018_MiniAODv2-v1/50000/26407FC9-AD2A-9F4A-9320-42868374071E.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018C-UL2018_MiniAODv2-v1/out_545.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/JetHT/MINIAOD/UL2018_MiniAODv2-v1/50000/F54E28D4-28CC-3540-A321-689D55B3C5B7.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018C-UL2018_MiniAODv2-v1/out_1018.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/JetHT/MINIAOD/UL2018_MiniAODv2-v1/50000/EB6C7F52-2666-7C4F-A320-94FD5812CF7F.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018C-UL2018_MiniAODv2-v1/out_548.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/JetHT/MINIAOD/UL2018_MiniAODv2-v1/50000/5B18346F-9444-8B42-BDA3-12E124CF4409.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018C-UL2018_MiniAODv2-v1/out_549.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/JetHT/MINIAOD/UL2018_MiniAODv2-v1/40000/FF2AEA9B-06BA-E24E-8F38-380FF296CF02.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018C-UL2018_MiniAODv2-v1/out_827.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/JetHT/MINIAOD/UL2018_MiniAODv2-v1/50000/D95EFF20-FA86-9A49-B305-88A853285B22.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018C-UL2018_MiniAODv2-v1/out_1019.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/JetHT/MINIAOD/UL2018_MiniAODv2-v1/50000/455788C1-4493-C141-918E-DCA33786F720.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018C-UL2018_MiniAODv2-v1/out_995.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/JetHT/MINIAOD/UL2018_MiniAODv2-v1/50000/953DAED3-96E2-B449-87CA-9FD1CFD5D541.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018C-UL2018_MiniAODv2-v1/out_994.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/JetHT/MINIAOD/UL2018_MiniAODv2-v1/50000/F8CC7E78-1F96-2545-8718-61A3E3AA9AB6.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018C-UL2018_MiniAODv2-v1/out_997.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/JetHT/MINIAOD/UL2018_MiniAODv2-v1/50000/0F8C717A-7F3C-EB44-A457-F64FAD8B9E5F.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018C-UL2018_MiniAODv2-v1/out_996.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/JetHT/MINIAOD/UL2018_MiniAODv2-v1/50000/A43E5D48-3AA4-E04F-9602-5750BBDD2409.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018C-UL2018_MiniAODv2-v1/out_991.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/JetHT/MINIAOD/UL2018_MiniAODv2-v1/50000/EB7AB93E-A6EA-2D4B-8D9F-37FD9B98663A.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018C-UL2018_MiniAODv2-v1/out_990.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/JetHT/MINIAOD/UL2018_MiniAODv2-v1/40000/856CF086-2BEE-164C-B713-F425015AA386.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018C-UL2018_MiniAODv2-v1/out_993.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/JetHT/MINIAOD/UL2018_MiniAODv2-v1/40000/88E8005A-30AB-5144-ABA4-68E9F21FFB5E.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018C-UL2018_MiniAODv2-v1/out_992.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/JetHT/MINIAOD/UL2018_MiniAODv2-v1/50000/6BEADF13-FAED-764A-A517-C503B8C246FF.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018C-UL2018_MiniAODv2-v1/out_999.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/JetHT/MINIAOD/UL2018_MiniAODv2-v1/50000/01C7584D-1471-7B4C-A29E-4E54076B2759.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018C-UL2018_MiniAODv2-v1/out_998.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/JetHT/MINIAOD/UL2018_MiniAODv2-v1/50000/E6B4A314-91EE-4947-ABB0-79207272F424.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018C-UL2018_MiniAODv2-v1/out_120.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/JetHT/MINIAOD/UL2018_MiniAODv2-v1/50000/637AF87E-0676-8E46-91D6-60A9D82C688E.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018C-UL2018_MiniAODv2-v1/out_121.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/JetHT/MINIAOD/UL2018_MiniAODv2-v1/50000/60F7DB59-A954-914A-BF9F-E013F1AC5508.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018C-UL2018_MiniAODv2-v1/out_122.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/JetHT/MINIAOD/UL2018_MiniAODv2-v1/50000/3F7F7976-95D2-A24C-85B4-F64F569873A5.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018C-UL2018_MiniAODv2-v1/out_123.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/JetHT/MINIAOD/UL2018_MiniAODv2-v1/50000/7ABAD414-4FBB-964F-A97D-AC40218E2988.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018C-UL2018_MiniAODv2-v1/out_124.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/JetHT/MINIAOD/UL2018_MiniAODv2-v1/50000/D8363549-4C1C-B645-9414-6A5EBCC7BA89.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018C-UL2018_MiniAODv2-v1/out_125.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/JetHT/MINIAOD/UL2018_MiniAODv2-v1/50000/6F5B121B-D3B7-DB4E-9E7B-8A3BE15F5083.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018C-UL2018_MiniAODv2-v1/out_126.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/JetHT/MINIAOD/UL2018_MiniAODv2-v1/50000/2CA5E191-2EE3-0945-A7FB-27B33986A249.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018C-UL2018_MiniAODv2-v1/out_127.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/JetHT/MINIAOD/UL2018_MiniAODv2-v1/50000/8E428122-D2BA-474F-99A5-0587502452DB.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018C-UL2018_MiniAODv2-v1/out_128.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/JetHT/MINIAOD/UL2018_MiniAODv2-v1/50000/BA7B637F-9E2C-2A4E-AF84-216CE58D1E7A.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018C-UL2018_MiniAODv2-v1/out_2.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/JetHT/MINIAOD/UL2018_MiniAODv2-v1/40000/89AB301F-3869-7848-8893-C1EB7419B148.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018C-UL2018_MiniAODv2-v1/out_219.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/JetHT/MINIAOD/UL2018_MiniAODv2-v1/40000/D417D7B7-B60C-D042-9A96-183979A74212.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018C-UL2018_MiniAODv2-v1/out_176.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/JetHT/MINIAOD/UL2018_MiniAODv2-v1/40000/BAB1F51C-8E13-E442-80A0-1297041FE379.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018C-UL2018_MiniAODv2-v1/out_214.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/JetHT/MINIAOD/UL2018_MiniAODv2-v1/50000/C3518616-F1AD-3646-A144-B5A0377EAC51.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018C-UL2018_MiniAODv2-v1/out_777.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/JetHT/MINIAOD/UL2018_MiniAODv2-v1/50000/2FB89E01-EF9E-D847-9429-77938070541C.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018C-UL2018_MiniAODv2-v1/out_1010.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/JetHT/MINIAOD/UL2018_MiniAODv2-v1/50000/F534687E-A60F-1940-9754-0240EE036B9B.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018C-UL2018_MiniAODv2-v1/out_563.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/JetHT/MINIAOD/UL2018_MiniAODv2-v1/50000/A67BDFD7-2476-5C4C-A39D-5964534E0E1D.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018C-UL2018_MiniAODv2-v1/out_1022.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/JetHT/MINIAOD/UL2018_MiniAODv2-v1/50000/DBA72070-AF65-DD46-A9F2-AC1AA93C99A3.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018C-UL2018_MiniAODv2-v1/out_1011.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/JetHT/MINIAOD/UL2018_MiniAODv2-v1/40000/E3A36704-6074-0D40-A463-1E3B982F4E02.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018C-UL2018_MiniAODv2-v1/out_928.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/JetHT/MINIAOD/UL2018_MiniAODv2-v1/40000/FB8C7389-7903-DA43-9C6A-FF7C439B0D1E.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018C-UL2018_MiniAODv2-v1/out_929.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/JetHT/MINIAOD/UL2018_MiniAODv2-v1/50000/1F8D6D2D-D18B-1642-A4C8-E28A5ADB0E15.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018C-UL2018_MiniAODv2-v1/out_416.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/JetHT/MINIAOD/UL2018_MiniAODv2-v1/50000/C053BAE0-58F6-6B43-9156-F1EBA24B1311.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018C-UL2018_MiniAODv2-v1/out_417.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/JetHT/MINIAOD/UL2018_MiniAODv2-v1/50000/47FD3604-E422-2049-B620-8DBA9E407BDD.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018C-UL2018_MiniAODv2-v1/out_410.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/JetHT/MINIAOD/UL2018_MiniAODv2-v1/50000/20ECF233-F2EF-8D46-AA88-864DA178DDCA.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018C-UL2018_MiniAODv2-v1/out_411.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/JetHT/MINIAOD/UL2018_MiniAODv2-v1/50000/A35A5F96-5C60-1E4D-85B9-06BD36866386.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018C-UL2018_MiniAODv2-v1/out_412.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/JetHT/MINIAOD/UL2018_MiniAODv2-v1/50000/A3C7EBF9-F5FB-1E4A-A7C2-675848D566EB.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018C-UL2018_MiniAODv2-v1/out_413.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/JetHT/MINIAOD/UL2018_MiniAODv2-v1/40000/C6E624AC-7A5D-A248-84C1-927F663EE502.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018C-UL2018_MiniAODv2-v1/out_920.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260000/02E34D99-4C69-5343-8765-986F0E150D59.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018C-UL2018_MiniAODv2-v1/out_498.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/JetHT/MINIAOD/UL2018_MiniAODv2-v1/40000/D7A39A88-DC67-704F-80F9-A603DAA7648E.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018C-UL2018_MiniAODv2-v1/out_922.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/JetHT/MINIAOD/UL2018_MiniAODv2-v1/40000/2B341A59-DA29-AE4B-829A-8EFC57DBBF55.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018C-UL2018_MiniAODv2-v1/out_923.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/JetHT/MINIAOD/UL2018_MiniAODv2-v1/40000/4087414F-D66E-854C-B819-29FB461853FE.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018C-UL2018_MiniAODv2-v1/out_924.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/JetHT/MINIAOD/UL2018_MiniAODv2-v1/40000/68FF0723-3779-F541-A64C-FB80AC23E26E.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018C-UL2018_MiniAODv2-v1/out_925.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/JetHT/MINIAOD/UL2018_MiniAODv2-v1/40000/8E625520-AF5A-0641-B6AA-AD7A2BABC2E5.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018C-UL2018_MiniAODv2-v1/out_926.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/JetHT/MINIAOD/UL2018_MiniAODv2-v1/40000/A383F958-B772-7642-99B7-F64B25F28FFA.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018C-UL2018_MiniAODv2-v1/out_927.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260000/5BD50005-6680-9043-B893-9498C613C21B.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018C-UL2018_MiniAODv2-v1/out_319.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260000/7C6C18B0-D9ED-154C-BA69-4319ECDDB0E8.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018C-UL2018_MiniAODv2-v1/out_318.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/JetHT/MINIAOD/UL2018_MiniAODv2-v1/50000/A0235352-F863-DB4F-9CD4-7C1B6FB2F0B3.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018C-UL2018_MiniAODv2-v1/out_313.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/JetHT/MINIAOD/UL2018_MiniAODv2-v1/40000/62266E35-D91A-2B43-B6B2-53776B9EAA02.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018C-UL2018_MiniAODv2-v1/out_312.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/JetHT/MINIAOD/UL2018_MiniAODv2-v1/40000/512E3E3D-AEE5-D243-A771-096C600B641B.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018C-UL2018_MiniAODv2-v1/out_311.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/JetHT/MINIAOD/UL2018_MiniAODv2-v1/40000/2D6C36CC-106B-2240-9DD3-95B26C8B61D8.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018C-UL2018_MiniAODv2-v1/out_310.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260000/B36930EE-54CA-2C4E-9149-82F3C30EEB92.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018C-UL2018_MiniAODv2-v1/out_317.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260000/A255CD86-8C17-9241-B72B-C0BC4AB14D7D.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018C-UL2018_MiniAODv2-v1/out_316.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/JetHT/MINIAOD/UL2018_MiniAODv2-v1/50000/33050F61-1D89-7547-B754-7410C3BAF08A.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018C-UL2018_MiniAODv2-v1/out_315.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/JetHT/MINIAOD/UL2018_MiniAODv2-v1/00000/FDF76AAE-1816-FC4A-AC4D-0F61A385420D.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018C-UL2018_MiniAODv2-v1/out_1183.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/JetHT/MINIAOD/UL2018_MiniAODv2-v1/40000/E7B592E2-C5E0-3144-9157-C1A0F18D4A98.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018C-UL2018_MiniAODv2-v1/out_921.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260000/B0388E29-AA0E-5446-9019-754BA6E0376D.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018C-UL2018_MiniAODv2-v1/out_496.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/JetHT/MINIAOD/UL2018_MiniAODv2-v1/40000/7FF34AC9-16D9-2244-8CC0-002E845EF803.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018C-UL2018_MiniAODv2-v1/out_1134.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/JetHT/MINIAOD/UL2018_MiniAODv2-v1/40000/0AA51B1F-3003-1D42-BD55-C94C3D5E8675.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018C-UL2018_MiniAODv2-v1/out_832.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/JetHT/MINIAOD/UL2018_MiniAODv2-v1/40000/C2772D97-3EBD-5C43-92E5-B61049B6E11D.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018C-UL2018_MiniAODv2-v1/out_833.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/JetHT/MINIAOD/UL2018_MiniAODv2-v1/40000/EA4DF19D-9452-DA4B-AF07-67E022A01BA3.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018C-UL2018_MiniAODv2-v1/out_830.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/JetHT/MINIAOD/UL2018_MiniAODv2-v1/40000/9FA7CA9F-EA0F-8243-BADC-FA09AB04FDCE.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018C-UL2018_MiniAODv2-v1/out_831.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/JetHT/MINIAOD/UL2018_MiniAODv2-v1/40000/AFD9C44F-E048-6C4E-9C5D-789894A1BD9C.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018C-UL2018_MiniAODv2-v1/out_836.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/JetHT/MINIAOD/UL2018_MiniAODv2-v1/40000/3F9195FD-06E4-C449-8A99-77C91DF79A6B.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018C-UL2018_MiniAODv2-v1/out_837.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/JetHT/MINIAOD/UL2018_MiniAODv2-v1/40000/66ECCAE1-11BD-2644-B33F-BD5A28132FB2.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018C-UL2018_MiniAODv2-v1/out_834.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/JetHT/MINIAOD/UL2018_MiniAODv2-v1/40000/321D76A1-0E7E-C14F-ACF9-76DF24AEF0D3.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018C-UL2018_MiniAODv2-v1/out_835.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/JetHT/MINIAOD/UL2018_MiniAODv2-v1/40000/2BBC8813-D283-FF44-B054-A72FDDD718B5.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018C-UL2018_MiniAODv2-v1/out_838.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/JetHT/MINIAOD/UL2018_MiniAODv2-v1/40000/3806D1D5-836F-7A4F-B30D-24975DEED2A2.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018C-UL2018_MiniAODv2-v1/out_839.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/JetHT/MINIAOD/UL2018_MiniAODv2-v1/50000/3530212B-CD15-A84F-B480-407D132DC037.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018C-UL2018_MiniAODv2-v1/out_3.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/JetHT/MINIAOD/UL2018_MiniAODv2-v1/50000/1F413ABE-2423-D546-80BF-246D5FD75A5B.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018C-UL2018_MiniAODv2-v1/out_725.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/JetHT/MINIAOD/UL2018_MiniAODv2-v1/50000/8675C6B0-B595-384C-BAF6-FF17AEA35C56.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018C-UL2018_MiniAODv2-v1/out_368.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/JetHT/MINIAOD/UL2018_MiniAODv2-v1/50000/75EC02A4-47B0-354B-8900-9DBB8F9382E1.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018C-UL2018_MiniAODv2-v1/out_369.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/JetHT/MINIAOD/UL2018_MiniAODv2-v1/50000/A483CE0A-2EFE-9746-91BA-4D873F135FA0.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018C-UL2018_MiniAODv2-v1/out_366.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/JetHT/MINIAOD/UL2018_MiniAODv2-v1/50000/0AE37F54-EC8F-5F44-B8B0-073D7E237BFB.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018C-UL2018_MiniAODv2-v1/out_367.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/JetHT/MINIAOD/UL2018_MiniAODv2-v1/50000/40FEEEBC-AEF0-D24D-B43A-04C6D3A4E097.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018C-UL2018_MiniAODv2-v1/out_364.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/JetHT/MINIAOD/UL2018_MiniAODv2-v1/50000/CE3736A2-66BA-FD4E-9540-3B2AB95313F4.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018C-UL2018_MiniAODv2-v1/out_365.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/JetHT/MINIAOD/UL2018_MiniAODv2-v1/50000/7E7FFAE5-7B46-244F-829B-5AA360AF3B96.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018C-UL2018_MiniAODv2-v1/out_362.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/JetHT/MINIAOD/UL2018_MiniAODv2-v1/50000/A516F7E8-65A4-4544-ACB8-B9E334678AA2.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018C-UL2018_MiniAODv2-v1/out_363.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/JetHT/MINIAOD/UL2018_MiniAODv2-v1/50000/0B8E5087-9512-9040-92CE-B5B12FB1AEE6.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018C-UL2018_MiniAODv2-v1/out_360.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/JetHT/MINIAOD/UL2018_MiniAODv2-v1/50000/13EA7835-3546-E445-8BFC-D19EFD9CCEA4.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018C-UL2018_MiniAODv2-v1/out_361.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/JetHT/MINIAOD/UL2018_MiniAODv2-v1/40000/73C0E045-5C9C-7046-B0EB-E3357334CEAD.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018C-UL2018_MiniAODv2-v1/out_218.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/JetHT/MINIAOD/UL2018_MiniAODv2-v1/50000/8E7A7DEE-66E7-754F-941A-A73F3D9DE29E.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018C-UL2018_MiniAODv2-v1/out_380.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/JetHT/MINIAOD/UL2018_MiniAODv2-v1/280000/C0B99939-2DF0-9E49-B2C3-2445F63B497B.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018C-UL2018_MiniAODv2-v1/out_861.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/JetHT/MINIAOD/UL2018_MiniAODv2-v1/50000/C45CC9B5-2FA5-CB40-AF1D-BAD4495EF32E.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018C-UL2018_MiniAODv2-v1/out_382.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/JetHT/MINIAOD/UL2018_MiniAODv2-v1/50000/DE07D16F-54E7-7242-9F31-6008D1B6CA27.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018C-UL2018_MiniAODv2-v1/out_383.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/JetHT/MINIAOD/UL2018_MiniAODv2-v1/50000/0942151E-7A14-4041-889A-03236C65653B.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018C-UL2018_MiniAODv2-v1/out_384.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/JetHT/MINIAOD/UL2018_MiniAODv2-v1/50000/49B99ACE-2085-4E4A-95C0-B54A405DB23D.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018C-UL2018_MiniAODv2-v1/out_385.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/JetHT/MINIAOD/UL2018_MiniAODv2-v1/50000/B6D8978A-859C-824A-99C7-2EAE4D1B9532.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018C-UL2018_MiniAODv2-v1/out_386.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/JetHT/MINIAOD/UL2018_MiniAODv2-v1/50000/10C36BA5-1558-2640-8887-41A64E126430.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018C-UL2018_MiniAODv2-v1/out_387.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/JetHT/MINIAOD/UL2018_MiniAODv2-v1/50000/C55334B6-3599-7248-87BB-51B54452780F.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018C-UL2018_MiniAODv2-v1/out_388.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/JetHT/MINIAOD/UL2018_MiniAODv2-v1/50000/08770EDE-23BF-A748-8AAC-065A6CF64F4A.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018C-UL2018_MiniAODv2-v1/out_389.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/JetHT/MINIAOD/UL2018_MiniAODv2-v1/50000/8C48C240-43E6-8741-B6E9-BE79B64443BE.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018C-UL2018_MiniAODv2-v1/out_784.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/JetHT/MINIAOD/UL2018_MiniAODv2-v1/50000/433711E0-4916-FC40-BE2E-D9EF1E2885AC.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018C-UL2018_MiniAODv2-v1/out_785.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/JetHT/MINIAOD/UL2018_MiniAODv2-v1/50000/03E080C7-BF9E-D140-ADA6-2AE520966B12.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018C-UL2018_MiniAODv2-v1/out_786.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/JetHT/MINIAOD/UL2018_MiniAODv2-v1/50000/6C6F26CA-9E63-2A48-9964-B1F9D6A6C9DD.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018C-UL2018_MiniAODv2-v1/out_787.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/JetHT/MINIAOD/UL2018_MiniAODv2-v1/50000/F3D1F56F-2571-A049-8E58-DB4CA6CF0D54.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018C-UL2018_MiniAODv2-v1/out_780.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/JetHT/MINIAOD/UL2018_MiniAODv2-v1/50000/A3A871DB-723E-1746-8BF4-433FAC285A5F.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018C-UL2018_MiniAODv2-v1/out_781.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/JetHT/MINIAOD/UL2018_MiniAODv2-v1/50000/1F2C4F6B-6C2A-B845-827C-F516E93D971A.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018C-UL2018_MiniAODv2-v1/out_782.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/JetHT/MINIAOD/UL2018_MiniAODv2-v1/50000/D3F34C5E-7798-1145-A78F-38E5633510A9.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018C-UL2018_MiniAODv2-v1/out_381.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/JetHT/MINIAOD/UL2018_MiniAODv2-v1/50000/4334CF0B-CE8F-734F-BCAC-E7F092260B33.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018C-UL2018_MiniAODv2-v1/out_788.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/JetHT/MINIAOD/UL2018_MiniAODv2-v1/50000/6F7544C1-BD74-7B45-B521-9A3AC4613C29.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018C-UL2018_MiniAODv2-v1/out_789.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/JetHT/MINIAOD/UL2018_MiniAODv2-v1/40000/4C103E7D-AACF-7A4B-B165-8010762A3A8B.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018C-UL2018_MiniAODv2-v1/out_1174.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/JetHT/MINIAOD/UL2018_MiniAODv2-v1/280000/D3BC6311-68E2-CD45-BC9A-4BC2084FFB89.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018C-UL2018_MiniAODv2-v1/out_860.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/JetHT/MINIAOD/UL2018_MiniAODv2-v1/410000/3342E1F3-97EF-634F-B6F0-CF55C333DB62.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018C-UL2018_MiniAODv2-v1/out_1164.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/JetHT/MINIAOD/UL2018_MiniAODv2-v1/50000/552C313F-20CD-614A-9387-1353EBBB8BF8.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018C-UL2018_MiniAODv2-v1/out_579.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/JetHT/MINIAOD/UL2018_MiniAODv2-v1/50000/FF1AD9F7-8A0E-9C44-B72A-ADC5031BB101.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018C-UL2018_MiniAODv2-v1/out_578.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/JetHT/MINIAOD/UL2018_MiniAODv2-v1/410000/CCF33A58-CA8E-F145-AB70-B072A34025CC.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018C-UL2018_MiniAODv2-v1/out_1165.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/JetHT/MINIAOD/UL2018_MiniAODv2-v1/50000/96C92C0F-AD68-2342-8549-37BD1DAE9C94.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018C-UL2018_MiniAODv2-v1/out_573.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/JetHT/MINIAOD/UL2018_MiniAODv2-v1/50000/44E1CB8B-09AC-6444-85A9-BB656CFD743F.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018C-UL2018_MiniAODv2-v1/out_572.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/JetHT/MINIAOD/UL2018_MiniAODv2-v1/50000/CCC469E6-9655-6B4F-AC1F-CC3CD58F6BCC.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018C-UL2018_MiniAODv2-v1/out_571.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/JetHT/MINIAOD/UL2018_MiniAODv2-v1/50000/7E4C2D3C-9D7B-5E46-8857-AD7E2B4833F3.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018C-UL2018_MiniAODv2-v1/out_570.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/JetHT/MINIAOD/UL2018_MiniAODv2-v1/50000/EE9AD10D-3536-3F4A-A903-2F6275F26A17.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018C-UL2018_MiniAODv2-v1/out_577.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/JetHT/MINIAOD/UL2018_MiniAODv2-v1/50000/D03BA901-A309-AB45-98F1-0EE722B39686.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018C-UL2018_MiniAODv2-v1/out_576.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/JetHT/MINIAOD/UL2018_MiniAODv2-v1/50000/7FA606E7-3CF0-A647-B4C5-47A5684EFCF0.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018C-UL2018_MiniAODv2-v1/out_575.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/JetHT/MINIAOD/UL2018_MiniAODv2-v1/50000/F86E0469-F938-D144-A627-2374BD150211.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018C-UL2018_MiniAODv2-v1/out_574.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/JetHT/MINIAOD/UL2018_MiniAODv2-v1/50000/1BB034BF-2384-3145-9C6E-816CABE3D14E.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018C-UL2018_MiniAODv2-v1/out_60.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/JetHT/MINIAOD/UL2018_MiniAODv2-v1/50000/902C0F0D-4FD7-2B4D-BF52-A4EEBFD5D716.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018C-UL2018_MiniAODv2-v1/out_61.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/JetHT/MINIAOD/UL2018_MiniAODv2-v1/50000/B4086420-A45D-9F4E-91BA-BBF0878EBEE2.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018C-UL2018_MiniAODv2-v1/out_62.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/JetHT/MINIAOD/UL2018_MiniAODv2-v1/280000/E800D587-89DE-4545-9AA6-0722342B556B.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018C-UL2018_MiniAODv2-v1/out_259.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/JetHT/MINIAOD/UL2018_MiniAODv2-v1/50000/C8507298-F734-6840-BCDA-6917C95BBAFC.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018C-UL2018_MiniAODv2-v1/out_64.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/JetHT/MINIAOD/UL2018_MiniAODv2-v1/50000/D5C9E037-25EB-A84C-AE07-4C7B844F8BB1.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018C-UL2018_MiniAODv2-v1/out_65.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/JetHT/MINIAOD/UL2018_MiniAODv2-v1/50000/3250E24B-C40A-914F-B05A-D5013BEF8B46.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018C-UL2018_MiniAODv2-v1/out_66.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/JetHT/MINIAOD/UL2018_MiniAODv2-v1/50000/EA0AC2D8-23A1-A649-A896-BB43CB105697.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018C-UL2018_MiniAODv2-v1/out_67.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/JetHT/MINIAOD/UL2018_MiniAODv2-v1/50000/CDBB328E-9E27-DA44-8A7E-8E6C0AD85C9A.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018C-UL2018_MiniAODv2-v1/out_68.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/JetHT/MINIAOD/UL2018_MiniAODv2-v1/280000/4E7EE0BC-9528-FC4B-95DA-19C146FAD681.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018C-UL2018_MiniAODv2-v1/out_253.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/JetHT/MINIAOD/UL2018_MiniAODv2-v1/40000/CE1579E2-B2EF-A340-B09E-F3D4A7CF28BE.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018C-UL2018_MiniAODv2-v1/out_250.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/JetHT/MINIAOD/UL2018_MiniAODv2-v1/40000/2CD2FEEC-9C42-6F4E-85FF-5CB68B03FEC4.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018C-UL2018_MiniAODv2-v1/out_251.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/JetHT/MINIAOD/UL2018_MiniAODv2-v1/280000/93A7F76A-7C29-B546-8C15-A3B718D078DE.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018C-UL2018_MiniAODv2-v1/out_256.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/JetHT/MINIAOD/UL2018_MiniAODv2-v1/280000/488EA50F-62C9-DB46-BD82-24B69498A3F1.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018C-UL2018_MiniAODv2-v1/out_257.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/JetHT/MINIAOD/UL2018_MiniAODv2-v1/280000/11891DCF-F418-3047-8BC2-A221BF2EE583.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018C-UL2018_MiniAODv2-v1/out_254.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/JetHT/MINIAOD/UL2018_MiniAODv2-v1/280000/CF575570-CE2B-2149-9471-F7D339002BA6.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018C-UL2018_MiniAODv2-v1/out_255.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260000/84D0F899-F474-3B41-A77F-9898E3BF16AA.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018C-UL2018_MiniAODv2-v1/out_499.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/JetHT/MINIAOD/UL2018_MiniAODv2-v1/50000/729C0005-30A2-174E-81CD-929EF7C94464.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018C-UL2018_MiniAODv2-v1/out_603.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/JetHT/MINIAOD/UL2018_MiniAODv2-v1/40000/331BD337-4B01-B644-91DC-FBCED707275D.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018C-UL2018_MiniAODv2-v1/out_1172.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/JetHT/MINIAOD/UL2018_MiniAODv2-v1/40000/56A4BBE7-2888-A54C-955A-4FC980B52CF8.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018C-UL2018_MiniAODv2-v1/out_156.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/JetHT/MINIAOD/UL2018_MiniAODv2-v1/40000/43874259-4921-ED4D-8803-7D82394CC275.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018C-UL2018_MiniAODv2-v1/out_939.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/JetHT/MINIAOD/UL2018_MiniAODv2-v1/50000/26620C25-345C-0043-A3B0-F07C17F0F66F.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018C-UL2018_MiniAODv2-v1/out_731.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/JetHT/MINIAOD/UL2018_MiniAODv2-v1/50000/88010485-3274-2D44-B1DE-8C4AC595A20C.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018C-UL2018_MiniAODv2-v1/out_730.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/JetHT/MINIAOD/UL2018_MiniAODv2-v1/50000/7ACE9D4D-3063-4342-BE10-79CFD7A7873B.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018C-UL2018_MiniAODv2-v1/out_733.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/JetHT/MINIAOD/UL2018_MiniAODv2-v1/40000/A2BF0D37-D967-5D43-A5AF-964DB2FD71C0.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018C-UL2018_MiniAODv2-v1/out_938.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/JetHT/MINIAOD/UL2018_MiniAODv2-v1/50000/8E70875F-996A-3249-B901-91C39C817209.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018C-UL2018_MiniAODv2-v1/out_735.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/JetHT/MINIAOD/UL2018_MiniAODv2-v1/50000/961825FB-994D-544C-99FD-F99C0E06C81A.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018C-UL2018_MiniAODv2-v1/out_734.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260000/5A8200F7-FB09-4440-86F5-2ED4D3FA8D0D.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018C-UL2018_MiniAODv2-v1/out_508.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/JetHT/MINIAOD/UL2018_MiniAODv2-v1/50000/02500728-E984-4343-B48F-CBB91501DA64.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018C-UL2018_MiniAODv2-v1/out_736.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/JetHT/MINIAOD/UL2018_MiniAODv2-v1/50000/7DA8DEE4-6418-E243-B79A-0007BA7983A6.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018C-UL2018_MiniAODv2-v1/out_1032.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/JetHT/MINIAOD/UL2018_MiniAODv2-v1/50000/4C2A0DA9-B9F9-2740-807E-C4F4E73A45F4.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018C-UL2018_MiniAODv2-v1/out_738.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/JetHT/MINIAOD/UL2018_MiniAODv2-v1/50000/6D3B4BF7-2FD9-F94C-AF1D-D128289AD47B.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018C-UL2018_MiniAODv2-v1/out_1030.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260000/E7666AF2-6B4D-1844-88D5-991F0D7AA42F.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018C-UL2018_MiniAODv2-v1/out_505.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260000/9AF98828-80C1-4144-939C-320321F4BED9.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018C-UL2018_MiniAODv2-v1/out_502.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260000/CEA317E4-7A65-6F46-8BB2-6A06BEE15AE3.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018C-UL2018_MiniAODv2-v1/out_503.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260000/B368FE63-D654-1E4B-81B6-E48DA96956F1.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018C-UL2018_MiniAODv2-v1/out_500.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260000/768943E4-BB93-D342-8B01-5261A31DF09F.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018C-UL2018_MiniAODv2-v1/out_501.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/JetHT/MINIAOD/UL2018_MiniAODv2-v1/50000/CE74ED29-9A26-8D44-A088-A2B020D49F56.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018C-UL2018_MiniAODv2-v1/out_630.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/JetHT/MINIAOD/UL2018_MiniAODv2-v1/50000/6CA8076D-F725-A64A-A6BF-15222D683873.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018C-UL2018_MiniAODv2-v1/out_631.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/JetHT/MINIAOD/UL2018_MiniAODv2-v1/50000/C90CACE5-BDB1-184E-975E-9461DB2164B8.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018C-UL2018_MiniAODv2-v1/out_632.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/JetHT/MINIAOD/UL2018_MiniAODv2-v1/50000/3D3862BD-13AF-844A-A1BB-C34855C5F920.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018C-UL2018_MiniAODv2-v1/out_633.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260000/5449BC3F-A287-E648-9136-1BFBF6E458C1.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018C-UL2018_MiniAODv2-v1/out_469.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260000/CD0E029F-CCF2-C842-93F8-B21FC31F26C0.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018C-UL2018_MiniAODv2-v1/out_468.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/JetHT/MINIAOD/UL2018_MiniAODv2-v1/50000/6F3FCA84-F89E-B144-B396-CB5A27E0B0E3.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018C-UL2018_MiniAODv2-v1/out_636.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/JetHT/MINIAOD/UL2018_MiniAODv2-v1/50000/E06388E2-7B3C-E44C-8422-F99E69D7E479.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018C-UL2018_MiniAODv2-v1/out_637.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260000/7BFFE783-6B29-F549-80CA-740918FBDE5A.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018C-UL2018_MiniAODv2-v1/out_465.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260000/FF75FB79-F9E5-894C-9469-E9AE2564E9F8.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018C-UL2018_MiniAODv2-v1/out_464.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/JetHT/MINIAOD/UL2018_MiniAODv2-v1/410000/326F56C1-9E30-0443-94AF-3313DEDDA459.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018C-UL2018_MiniAODv2-v1/out_1128.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260000/71B07C78-44C7-AB48-A7EB-C274617B6B8C.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018C-UL2018_MiniAODv2-v1/out_466.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260000/C16A88D4-DD6D-B445-A65E-2DA22D1C8067.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018C-UL2018_MiniAODv2-v1/out_461.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/JetHT/MINIAOD/UL2018_MiniAODv2-v1/280000/690C5AD3-6EBA-0B4E-8652-BB689296CEB1.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018C-UL2018_MiniAODv2-v1/out_900.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260000/EF07BA2C-094C-524C-8756-3DF1A992DA00.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018C-UL2018_MiniAODv2-v1/out_463.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260000/AF3D0F8C-4FEF-E944-AE48-81A443A0CCD8.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018C-UL2018_MiniAODv2-v1/out_462.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/JetHT/MINIAOD/UL2018_MiniAODv2-v1/50000/BC03F212-EA3E-504E-BFFC-182D4A5F1DD5.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018C-UL2018_MiniAODv2-v1/out_281.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/JetHT/MINIAOD/UL2018_MiniAODv2-v1/280000/4E7D9EB5-D3D3-DE4E-B8FF-E4FD25071686.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018C-UL2018_MiniAODv2-v1/out_901.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/JetHT/MINIAOD/UL2018_MiniAODv2-v1/40000/9BD231A3-0747-6E44-95E6-0C5079C18E20.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018C-UL2018_MiniAODv2-v1/out_168.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/JetHT/MINIAOD/UL2018_MiniAODv2-v1/40000/F0EA0A74-FC0D-0C44-9205-0432CA36F7DF.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018C-UL2018_MiniAODv2-v1/out_169.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/JetHT/MINIAOD/UL2018_MiniAODv2-v1/40000/1A6883F1-2D6C-984A-BDDA-298FA5F9AE55.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018C-UL2018_MiniAODv2-v1/out_164.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/JetHT/MINIAOD/UL2018_MiniAODv2-v1/40000/584057E9-0FA5-FC44-8868-58BDCCBB63DB.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018C-UL2018_MiniAODv2-v1/out_165.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/JetHT/MINIAOD/UL2018_MiniAODv2-v1/40000/642E2542-38A6-6A41-8983-7B5B1E57C118.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018C-UL2018_MiniAODv2-v1/out_166.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/JetHT/MINIAOD/UL2018_MiniAODv2-v1/40000/0B2A0D7F-4652-1242-8F91-25A8BF01F14A.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018C-UL2018_MiniAODv2-v1/out_167.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/JetHT/MINIAOD/UL2018_MiniAODv2-v1/40000/3F0BBA51-E915-5C4B-BAE4-66EA0EED026C.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018C-UL2018_MiniAODv2-v1/out_160.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/JetHT/MINIAOD/UL2018_MiniAODv2-v1/40000/E0D84D13-014C-9848-88C4-98A06842AAD0.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018C-UL2018_MiniAODv2-v1/out_161.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/JetHT/MINIAOD/UL2018_MiniAODv2-v1/50000/A301D456-2619-CB4E-8ED0-CA0599F3D6A0.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018C-UL2018_MiniAODv2-v1/out_1016.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/JetHT/MINIAOD/UL2018_MiniAODv2-v1/40000/727D054F-692B-0243-AE7A-C126BF209309.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018C-UL2018_MiniAODv2-v1/out_163.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/JetHT/MINIAOD/UL2018_MiniAODv2-v1/50000/7424534B-32CB-AB45-9BA2-305C6F9DBB72.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018C-UL2018_MiniAODv2-v1/out_964.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/JetHT/MINIAOD/UL2018_MiniAODv2-v1/50000/17E88C85-576F-A247-A82A-F9703E9C3A15.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018C-UL2018_MiniAODv2-v1/out_965.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/JetHT/MINIAOD/UL2018_MiniAODv2-v1/50000/F526EA7F-B745-8240-BE9C-B06AA4B0AAB9.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018C-UL2018_MiniAODv2-v1/out_966.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/JetHT/MINIAOD/UL2018_MiniAODv2-v1/50000/6044CF82-A29F-8949-9E10-73E6D189F602.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018C-UL2018_MiniAODv2-v1/out_967.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/JetHT/MINIAOD/UL2018_MiniAODv2-v1/50000/22F6F811-7BEA-5041-8356-8442163A5B14.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018C-UL2018_MiniAODv2-v1/out_960.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/JetHT/MINIAOD/UL2018_MiniAODv2-v1/50000/DFDACFB9-1A08-F34C-A9A8-FC781844697A.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018C-UL2018_MiniAODv2-v1/out_961.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/JetHT/MINIAOD/UL2018_MiniAODv2-v1/50000/EC48F5E6-82A1-7049-8260-95EE8FB269DD.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018C-UL2018_MiniAODv2-v1/out_962.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/JetHT/MINIAOD/UL2018_MiniAODv2-v1/50000/3C3403F6-CEE4-5248-BBFE-4A07028B9EC3.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018C-UL2018_MiniAODv2-v1/out_963.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/JetHT/MINIAOD/UL2018_MiniAODv2-v1/410000/FFC5BABF-00B1-2641-83CA-3E491781690E.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018C-UL2018_MiniAODv2-v1/out_1129.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/JetHT/MINIAOD/UL2018_MiniAODv2-v1/50000/659C6DCD-E209-B341-B43F-6395DE1C71B4.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018C-UL2018_MiniAODv2-v1/out_1039.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/JetHT/MINIAOD/UL2018_MiniAODv2-v1/50000/CA656FDB-CEF1-BD40-8BA4-F0F4FF5BCC87.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018C-UL2018_MiniAODv2-v1/out_783.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/JetHT/MINIAOD/UL2018_MiniAODv2-v1/50000/303BFD0F-3665-6249-B0A6-D17B904BEC18.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018C-UL2018_MiniAODv2-v1/out_968.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/JetHT/MINIAOD/UL2018_MiniAODv2-v1/50000/F20E21DD-2912-174C-B1FB-B5340DBFC18F.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018C-UL2018_MiniAODv2-v1/out_969.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/JetHT/MINIAOD/UL2018_MiniAODv2-v1/40000/0D10678E-DD4F-BA4C-A8DD-5DB222A7AC4A.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018C-UL2018_MiniAODv2-v1/out_936.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/JetHT/MINIAOD/UL2018_MiniAODv2-v1/410000/C0439F17-9521-AA49-BC1C-DA0E09292837.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018C-UL2018_MiniAODv2-v1/out_1106.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/JetHT/MINIAOD/UL2018_MiniAODv2-v1/40000/5A483013-0D43-434C-8ECB-6B9DD1728358.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018C-UL2018_MiniAODv2-v1/out_1107.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/JetHT/MINIAOD/UL2018_MiniAODv2-v1/40000/D40055F8-B260-BB48-B0A2-6421EB5DA08F.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018C-UL2018_MiniAODv2-v1/out_1104.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/JetHT/MINIAOD/UL2018_MiniAODv2-v1/410000/8B768AAA-9472-DD47-820D-0CA7FE7F6749.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018C-UL2018_MiniAODv2-v1/out_1105.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/JetHT/MINIAOD/UL2018_MiniAODv2-v1/410000/3DA5733F-BB7F-814A-B1C6-9459663D4BBA.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018C-UL2018_MiniAODv2-v1/out_1102.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/JetHT/MINIAOD/UL2018_MiniAODv2-v1/410000/C6CF337A-6FB9-E94D-B172-122F3A8F566B.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018C-UL2018_MiniAODv2-v1/out_1103.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/JetHT/MINIAOD/UL2018_MiniAODv2-v1/40000/2A412C2D-FC65-8E42-B49A-9BFBCB83F0AE.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018C-UL2018_MiniAODv2-v1/out_1100.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/JetHT/MINIAOD/UL2018_MiniAODv2-v1/40000/CB82C2C8-0AC5-514A-9694-9EF90A0CAC7C.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018C-UL2018_MiniAODv2-v1/out_1101.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/JetHT/MINIAOD/UL2018_MiniAODv2-v1/40000/45EF1DFB-BE24-9C42-AB31-9427B268350F.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018C-UL2018_MiniAODv2-v1/out_934.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/JetHT/MINIAOD/UL2018_MiniAODv2-v1/280000/C1BF4DBD-E10D-5448-A865-C87CEAD41405.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018C-UL2018_MiniAODv2-v1/out_908.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/JetHT/MINIAOD/UL2018_MiniAODv2-v1/40000/4C8C4895-A2A6-F741-8D61-6FAE5710758C.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018C-UL2018_MiniAODv2-v1/out_1108.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/JetHT/MINIAOD/UL2018_MiniAODv2-v1/40000/2ED75D34-A6AF-1849-985A-4EC39342ADC3.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018C-UL2018_MiniAODv2-v1/out_1109.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/JetHT/MINIAOD/UL2018_MiniAODv2-v1/50000/08045EF2-1F59-F84E-93A3-2137B0C226C3.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018C-UL2018_MiniAODv2-v1/out_670.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/JetHT/MINIAOD/UL2018_MiniAODv2-v1/280000/848346FD-CDD5-9945-BC57-574AF5671C52.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018C-UL2018_MiniAODv2-v1/out_909.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/JetHT/MINIAOD/UL2018_MiniAODv2-v1/50000/16BD5D76-224F-0C4D-B966-AD51D4DBD231.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018C-UL2018_MiniAODv2-v1/out_722.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/JetHT/MINIAOD/UL2018_MiniAODv2-v1/50000/71E85344-22A0-FF4C-8800-30FBA38FFBFE.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018C-UL2018_MiniAODv2-v1/out_584.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/JetHT/MINIAOD/UL2018_MiniAODv2-v1/50000/F4AAF56C-74ED-F04E-921A-816AC5CE6CCA.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018C-UL2018_MiniAODv2-v1/out_678.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/JetHT/MINIAOD/UL2018_MiniAODv2-v1/50000/70FD6CE1-8DC1-1A48-AE6D-D6359E70CC8A.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018C-UL2018_MiniAODv2-v1/out_609.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/JetHT/MINIAOD/UL2018_MiniAODv2-v1/410000/16971212-E7D1-9E49-9518-DE3F9E682297.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018C-UL2018_MiniAODv2-v1/out_1158.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/JetHT/MINIAOD/UL2018_MiniAODv2-v1/50000/DC9E4F63-1009-C342-9E33-7B3BEF212796.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018C-UL2018_MiniAODv2-v1/out_426.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/JetHT/MINIAOD/UL2018_MiniAODv2-v1/280000/A41ADCE9-B7FE-094D-9F52-D85C65EC6E2E.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018C-UL2018_MiniAODv2-v1/out_878.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/JetHT/MINIAOD/UL2018_MiniAODv2-v1/280000/AED0238E-9B13-B548-B061-1909F180FA17.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018C-UL2018_MiniAODv2-v1/out_879.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/JetHT/MINIAOD/UL2018_MiniAODv2-v1/280000/CD08E02D-68C7-B448-BB17-EA47DE62737B.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018C-UL2018_MiniAODv2-v1/out_876.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/JetHT/MINIAOD/UL2018_MiniAODv2-v1/280000/6F9347EE-103E-394B-AF03-28EEA590254D.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018C-UL2018_MiniAODv2-v1/out_877.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/JetHT/MINIAOD/UL2018_MiniAODv2-v1/280000/340A9F13-275B-3A45-8BF2-A8DB93295F94.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018C-UL2018_MiniAODv2-v1/out_874.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/JetHT/MINIAOD/UL2018_MiniAODv2-v1/280000/B2B8B5C5-5A67-0E46-A1F0-70137895320C.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018C-UL2018_MiniAODv2-v1/out_875.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/JetHT/MINIAOD/UL2018_MiniAODv2-v1/280000/7147E380-234C-684F-A228-98226E6D3575.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018C-UL2018_MiniAODv2-v1/out_872.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/JetHT/MINIAOD/UL2018_MiniAODv2-v1/280000/81BE2B40-3FE5-1B43-BDAB-018E4178D49D.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018C-UL2018_MiniAODv2-v1/out_873.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/JetHT/MINIAOD/UL2018_MiniAODv2-v1/280000/BECF1F6E-E88A-F94A-9F15-48927A4AC0D4.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018C-UL2018_MiniAODv2-v1/out_870.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/JetHT/MINIAOD/UL2018_MiniAODv2-v1/280000/089D5609-DA62-3D41-BCD7-B03113B71B80.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018C-UL2018_MiniAODv2-v1/out_871.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/JetHT/MINIAOD/UL2018_MiniAODv2-v1/50000/E7F6FD60-1860-3C4A-B76A-76757E2868DD.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018C-UL2018_MiniAODv2-v1/out_674.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/JetHT/MINIAOD/UL2018_MiniAODv2-v1/50000/0A17ABCD-2DFE-6B46-9124-34096F6F5EBC.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018C-UL2018_MiniAODv2-v1/out_675.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/JetHT/MINIAOD/UL2018_MiniAODv2-v1/50000/F1FE18B7-8081-4F45-B79F-5836B195E97A.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018C-UL2018_MiniAODv2-v1/out_9.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/JetHT/MINIAOD/UL2018_MiniAODv2-v1/50000/8BB2D6DE-DE99-7D4E-A9A9-AC3BA651D12C.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018C-UL2018_MiniAODv2-v1/out_676.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/JetHT/MINIAOD/UL2018_MiniAODv2-v1/280000/9F731F2D-F35D-064C-905B-6F5D20823C34.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018C-UL2018_MiniAODv2-v1/out_890.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/JetHT/MINIAOD/UL2018_MiniAODv2-v1/280000/442C29F0-305A-9C40-A9A9-2B28E8BAF5C6.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018C-UL2018_MiniAODv2-v1/out_891.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/JetHT/MINIAOD/UL2018_MiniAODv2-v1/280000/17988D3A-A75D-2243-98BB-E68D806CF113.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018C-UL2018_MiniAODv2-v1/out_892.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/JetHT/MINIAOD/UL2018_MiniAODv2-v1/280000/B5B0F147-69E0-0C47-BE54-CA2C9CD8FDF6.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018C-UL2018_MiniAODv2-v1/out_893.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/JetHT/MINIAOD/UL2018_MiniAODv2-v1/280000/0D3585D2-BB0B-D24E-9890-4A1FCDA1A8FB.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018C-UL2018_MiniAODv2-v1/out_894.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/JetHT/MINIAOD/UL2018_MiniAODv2-v1/50000/C5391DF2-3522-274F-B5F0-DB8021ABA63E.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018C-UL2018_MiniAODv2-v1/out_647.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/JetHT/MINIAOD/UL2018_MiniAODv2-v1/280000/6A592EEF-9C26-4F4F-9574-484BC992F4E7.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018C-UL2018_MiniAODv2-v1/out_896.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/JetHT/MINIAOD/UL2018_MiniAODv2-v1/280000/04760B9F-0470-1D4B-81EB-666F0B2DD3DA.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018C-UL2018_MiniAODv2-v1/out_897.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/JetHT/MINIAOD/UL2018_MiniAODv2-v1/280000/4FAD5241-F055-BC42-921B-0BCE54EA6E1E.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018C-UL2018_MiniAODv2-v1/out_898.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/JetHT/MINIAOD/UL2018_MiniAODv2-v1/280000/784B216A-95FA-DB4B-ABFD-F450427B4F67.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018C-UL2018_MiniAODv2-v1/out_899.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/JetHT/MINIAOD/UL2018_MiniAODv2-v1/50000/7EA85237-DFAA-0946-8C43-E12D476C0F80.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018C-UL2018_MiniAODv2-v1/out_1087.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/JetHT/MINIAOD/UL2018_MiniAODv2-v1/50000/F2D737ED-B672-C242-85D7-FE764CCEA894.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018C-UL2018_MiniAODv2-v1/out_646.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/JetHT/MINIAOD/UL2018_MiniAODv2-v1/40000/A05CB805-D209-D443-83B9-334D099980CF.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018C-UL2018_MiniAODv2-v1/out_1099.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/JetHT/MINIAOD/UL2018_MiniAODv2-v1/410000/D38AF816-68BA-5046-9EF3-80389A5EC135.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018C-UL2018_MiniAODv2-v1/out_1138.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/JetHT/MINIAOD/UL2018_MiniAODv2-v1/50000/BC81D5A2-24AB-0A4C-BDD2-0CC0874FF15A.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018C-UL2018_MiniAODv2-v1/out_672.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/JetHT/MINIAOD/UL2018_MiniAODv2-v1/50000/0AF30B59-C8F3-4345-ACE9-FEDA5A12D0E5.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018C-UL2018_MiniAODv2-v1/out_648.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/JetHT/MINIAOD/UL2018_MiniAODv2-v1/50000/F4290153-04D5-8348-A6AA-B69755FC36B9.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018C-UL2018_MiniAODv2-v1/out_673.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/JetHT/MINIAOD/UL2018_MiniAODv2-v1/50000/3F8E421E-EE1C-F647-9663-071DDB4B7E67.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018C-UL2018_MiniAODv2-v1/out_1036.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/JetHT/MINIAOD/UL2018_MiniAODv2-v1/50000/A9258FB2-96E0-4542-B157-A072B983D4B2.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018C-UL2018_MiniAODv2-v1/out_638.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/JetHT/MINIAOD/UL2018_MiniAODv2-v1/410000/EE0AF87C-61ED-7A47-BB7C-35022579B7AE.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018C-UL2018_MiniAODv2-v1/out_1173.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/JetHT/MINIAOD/UL2018_MiniAODv2-v1/50000/291BEF52-A5AC-2B46-BCB1-22E0880F01EF.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018C-UL2018_MiniAODv2-v1/out_1086.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/JetHT/MINIAOD/UL2018_MiniAODv2-v1/50000/355D69CD-DA7B-8A4D-8286-70FA62F278E5.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018C-UL2018_MiniAODv2-v1/out_357.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/JetHT/MINIAOD/UL2018_MiniAODv2-v1/50000/6A578BDF-8CFA-CF44-AAA8-A66EDB0545E2.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018C-UL2018_MiniAODv2-v1/out_356.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/JetHT/MINIAOD/UL2018_MiniAODv2-v1/40000/D72DC8D1-3DEA-1B43-807A-61004B0E872E.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018C-UL2018_MiniAODv2-v1/out_809.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/JetHT/MINIAOD/UL2018_MiniAODv2-v1/40000/96E77F0D-7AB0-3048-910F-8709B0628FEE.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018C-UL2018_MiniAODv2-v1/out_808.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/JetHT/MINIAOD/UL2018_MiniAODv2-v1/50000/AE0CD33A-2C29-E447-BE94-29A586E681B7.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018C-UL2018_MiniAODv2-v1/out_353.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/JetHT/MINIAOD/UL2018_MiniAODv2-v1/50000/A78957F7-FB66-2647-A885-2EF57859D98D.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018C-UL2018_MiniAODv2-v1/out_352.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/JetHT/MINIAOD/UL2018_MiniAODv2-v1/50000/35BB1891-A07D-4340-BF67-D0BCD42198AB.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018C-UL2018_MiniAODv2-v1/out_351.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/JetHT/MINIAOD/UL2018_MiniAODv2-v1/50000/99443738-48AA-3B47-8A65-1E5D1C3778E1.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018C-UL2018_MiniAODv2-v1/out_350.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/JetHT/MINIAOD/UL2018_MiniAODv2-v1/40000/2623CD99-647A-7344-A8E3-E969F980C3CD.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018C-UL2018_MiniAODv2-v1/out_803.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/JetHT/MINIAOD/UL2018_MiniAODv2-v1/40000/25C97E8A-6785-E44C-9D31-03D35AC3AA25.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018C-UL2018_MiniAODv2-v1/out_802.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/JetHT/MINIAOD/UL2018_MiniAODv2-v1/40000/24CCB7EA-CFD1-1749-9013-D4D6F34FAA76.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018C-UL2018_MiniAODv2-v1/out_801.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/JetHT/MINIAOD/UL2018_MiniAODv2-v1/40000/6C354F48-F7A7-4548-A337-8F4E8CB8BCF7.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018C-UL2018_MiniAODv2-v1/out_800.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/JetHT/MINIAOD/UL2018_MiniAODv2-v1/40000/1FDF802A-6C88-B442-B293-21F063116B21.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018C-UL2018_MiniAODv2-v1/out_807.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/JetHT/MINIAOD/UL2018_MiniAODv2-v1/40000/AF60DC5A-3E55-6449-8822-49C53A465788.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018C-UL2018_MiniAODv2-v1/out_806.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/JetHT/MINIAOD/UL2018_MiniAODv2-v1/40000/2807F6A9-A392-E941-86F1-AB67A6EE0C27.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018C-UL2018_MiniAODv2-v1/out_805.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/JetHT/MINIAOD/UL2018_MiniAODv2-v1/40000/7BD62993-D4E2-B043-8FC1-34CEC6AAD269.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018C-UL2018_MiniAODv2-v1/out_804.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/JetHT/MINIAOD/UL2018_MiniAODv2-v1/40000/7A501B50-46EE-DE44-A4A8-ECA6EDBF47F6.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018C-UL2018_MiniAODv2-v1/out_216.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/JetHT/MINIAOD/UL2018_MiniAODv2-v1/40000/A5E64199-F386-664A-BCE4-8B73CA051CDD.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018C-UL2018_MiniAODv2-v1/out_217.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/JetHT/MINIAOD/UL2018_MiniAODv2-v1/50000/8ACBC8EA-BD37-2448-B0B4-56245288CFD4.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018C-UL2018_MiniAODv2-v1/out_768.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/JetHT/MINIAOD/UL2018_MiniAODv2-v1/50000/B553D79F-DA22-F040-9710-0925BDC5E3B6.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018C-UL2018_MiniAODv2-v1/out_769.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/JetHT/MINIAOD/UL2018_MiniAODv2-v1/40000/4178E7FD-DE6F-714C-989A-32037C3649CC.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018C-UL2018_MiniAODv2-v1/out_212.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/JetHT/MINIAOD/UL2018_MiniAODv2-v1/40000/A4C41ABB-0B9A-A945-90EE-FD952B9813BC.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018C-UL2018_MiniAODv2-v1/out_213.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/JetHT/MINIAOD/UL2018_MiniAODv2-v1/40000/B6FD2C50-6AE6-B648-BB85-2C2EDF74EE71.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018C-UL2018_MiniAODv2-v1/out_210.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/JetHT/MINIAOD/UL2018_MiniAODv2-v1/40000/A8C1FA1F-F4C9-854C-B601-D5158C751203.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018C-UL2018_MiniAODv2-v1/out_211.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/JetHT/MINIAOD/UL2018_MiniAODv2-v1/50000/D6661C10-EF1F-114E-BA1A-E2273302F74E.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018C-UL2018_MiniAODv2-v1/out_762.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/JetHT/MINIAOD/UL2018_MiniAODv2-v1/50000/97CBF126-5634-DF4A-8681-FF5B466E853F.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018C-UL2018_MiniAODv2-v1/out_763.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/JetHT/MINIAOD/UL2018_MiniAODv2-v1/50000/DF912F18-67D2-3D4A-832B-020838068678.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018C-UL2018_MiniAODv2-v1/out_760.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/JetHT/MINIAOD/UL2018_MiniAODv2-v1/50000/CD6C4260-E40C-B44C-9C0E-F8CB768A9F27.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018C-UL2018_MiniAODv2-v1/out_761.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/JetHT/MINIAOD/UL2018_MiniAODv2-v1/50000/6FDB16AE-8DF1-1F44-923B-4937CA0ADA20.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018C-UL2018_MiniAODv2-v1/out_766.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/JetHT/MINIAOD/UL2018_MiniAODv2-v1/50000/1887786D-02E0-E24B-B4D5-6AE0E616D8CB.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018C-UL2018_MiniAODv2-v1/out_767.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/JetHT/MINIAOD/UL2018_MiniAODv2-v1/50000/0123D692-87DA-AF48-B6E2-3C2998B2E156.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018C-UL2018_MiniAODv2-v1/out_764.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/JetHT/MINIAOD/UL2018_MiniAODv2-v1/50000/0B6AE7C4-19A1-7B47-BC2D-B37C89C23723.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018C-UL2018_MiniAODv2-v1/out_765.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/JetHT/MINIAOD/UL2018_MiniAODv2-v1/280000/800A2086-127C-D84F-9686-558DFCC72327.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018C-UL2018_MiniAODv2-v1/out_40.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/JetHT/MINIAOD/UL2018_MiniAODv2-v1/50000/BEE41D72-D5B1-2044-96BB-E8C1DF18B027.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018C-UL2018_MiniAODv2-v1/out_41.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/JetHT/MINIAOD/UL2018_MiniAODv2-v1/50000/7E0AF278-D1DB-6D4D-8529-846C206E8BE5.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018C-UL2018_MiniAODv2-v1/out_1033.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/JetHT/MINIAOD/UL2018_MiniAODv2-v1/410000/8D0FE3DD-A0BC-A44C-9ED1-CB51B92F4058.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018C-UL2018_MiniAODv2-v1/out_1119.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/JetHT/MINIAOD/UL2018_MiniAODv2-v1/50000/F74924A7-62BE-534A-B6DF-063AC31B400D.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018C-UL2018_MiniAODv2-v1/out_1078.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/JetHT/MINIAOD/UL2018_MiniAODv2-v1/50000/C8EFD23A-48F2-5949-B0F5-D5421C4D123C.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018C-UL2018_MiniAODv2-v1/out_1079.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/JetHT/MINIAOD/UL2018_MiniAODv2-v1/50000/293E1CFC-D1EF-2D4E-8B92-DD13A1C7FEE6.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018C-UL2018_MiniAODv2-v1/out_1076.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/JetHT/MINIAOD/UL2018_MiniAODv2-v1/50000/B3EF9444-932E-6A4D-B9C1-4CC961B2DA93.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018C-UL2018_MiniAODv2-v1/out_1077.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/JetHT/MINIAOD/UL2018_MiniAODv2-v1/50000/95DBCDC3-B169-814F-8032-36FDFE0ED087.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018C-UL2018_MiniAODv2-v1/out_1074.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/JetHT/MINIAOD/UL2018_MiniAODv2-v1/50000/ACA0F121-00D2-F04C-AFDB-06AC046145C8.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018C-UL2018_MiniAODv2-v1/out_1075.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/JetHT/MINIAOD/UL2018_MiniAODv2-v1/50000/0ED1CCF0-3971-2247-8794-4846C1FFEFC3.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018C-UL2018_MiniAODv2-v1/out_1072.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/JetHT/MINIAOD/UL2018_MiniAODv2-v1/50000/E043BB39-8922-A248-8A5A-652B6F07B644.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018C-UL2018_MiniAODv2-v1/out_1073.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/JetHT/MINIAOD/UL2018_MiniAODv2-v1/50000/1BD11271-F602-FB4F-82F1-72E70C1B0216.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018C-UL2018_MiniAODv2-v1/out_1070.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/JetHT/MINIAOD/UL2018_MiniAODv2-v1/50000/3C49C809-5EE2-B049-AD28-431FBCD7CDF2.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018C-UL2018_MiniAODv2-v1/out_1071.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/JetHT/MINIAOD/UL2018_MiniAODv2-v1/50000/16A81C1E-A05C-2544-B01A-ADE018CA2F6F.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018C-UL2018_MiniAODv2-v1/out_289.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/JetHT/MINIAOD/UL2018_MiniAODv2-v1/50000/80413976-7B91-C445-96C8-3820FBFD513E.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018C-UL2018_MiniAODv2-v1/out_288.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/JetHT/MINIAOD/UL2018_MiniAODv2-v1/50000/9608943E-A066-4647-A043-880B9EB935C6.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018C-UL2018_MiniAODv2-v1/out_1003.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/JetHT/MINIAOD/UL2018_MiniAODv2-v1/50000/A3176290-18E4-5D40-A354-15E1C307DDB6.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018C-UL2018_MiniAODv2-v1/out_4.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/JetHT/MINIAOD/UL2018_MiniAODv2-v1/50000/8C9323F1-CFA0-9647-97FC-66948D26DD5B.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018C-UL2018_MiniAODv2-v1/out_1179.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/JetHT/MINIAOD/UL2018_MiniAODv2-v1/280000/49A70ACE-FFF6-0C47-8D9C-FE07AAB973C9.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018C-UL2018_MiniAODv2-v1/out_280.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/JetHT/MINIAOD/UL2018_MiniAODv2-v1/50000/3258BBC8-8774-8F40-A35A-7ADBD2327576.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018C-UL2018_MiniAODv2-v1/out_283.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/JetHT/MINIAOD/UL2018_MiniAODv2-v1/50000/7F394EA3-A97F-9B43-92CF-983D69A56F14.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018C-UL2018_MiniAODv2-v1/out_282.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/JetHT/MINIAOD/UL2018_MiniAODv2-v1/50000/91F15CBE-9932-394B-8B86-C2F2B094066C.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018C-UL2018_MiniAODv2-v1/out_285.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/JetHT/MINIAOD/UL2018_MiniAODv2-v1/50000/4A80151E-2843-D342-803E-B604A4863770.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018C-UL2018_MiniAODv2-v1/out_284.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/JetHT/MINIAOD/UL2018_MiniAODv2-v1/50000/D6815FC6-2C19-844E-AB2F-4D5A82F32C24.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018C-UL2018_MiniAODv2-v1/out_287.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/JetHT/MINIAOD/UL2018_MiniAODv2-v1/50000/5518A2F0-73D7-DA43-95AB-F8B66B643BDB.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018C-UL2018_MiniAODv2-v1/out_286.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/JetHT/MINIAOD/UL2018_MiniAODv2-v1/410000/4B327B50-CD3C-F047-A281-882C9DDF0A94.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018C-UL2018_MiniAODv2-v1/out_1094.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/JetHT/MINIAOD/UL2018_MiniAODv2-v1/410000/5DE34219-37AA-5249-B07B-CF7ECF808E43.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018C-UL2018_MiniAODv2-v1/out_1095.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/JetHT/MINIAOD/UL2018_MiniAODv2-v1/410000/055398F6-EA6B-4146-BDCC-EAA8AA4FDBFA.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018C-UL2018_MiniAODv2-v1/out_1096.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/JetHT/MINIAOD/UL2018_MiniAODv2-v1/410000/12CBFA03-9481-E44C-B8BC-162EA0B38590.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018C-UL2018_MiniAODv2-v1/out_1097.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/JetHT/MINIAOD/UL2018_MiniAODv2-v1/410000/2C801162-7E8D-9F4F-AEAD-BE0BA617E96D.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018C-UL2018_MiniAODv2-v1/out_1090.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/JetHT/MINIAOD/UL2018_MiniAODv2-v1/410000/A941ABD7-37F7-6B40-A458-184DD9686E4C.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018C-UL2018_MiniAODv2-v1/out_1091.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/JetHT/MINIAOD/UL2018_MiniAODv2-v1/410000/38384553-4AA3-6249-88AB-2E524F7F5FE0.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018C-UL2018_MiniAODv2-v1/out_1092.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/JetHT/MINIAOD/UL2018_MiniAODv2-v1/410000/9119A4E5-5438-6F46-A8D5-FD88C39A4427.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018C-UL2018_MiniAODv2-v1/out_1093.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/JetHT/MINIAOD/UL2018_MiniAODv2-v1/410000/291D3B24-4788-D448-B2DF-D1608D9EFDEC.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018C-UL2018_MiniAODv2-v1/out_1155.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/JetHT/MINIAOD/UL2018_MiniAODv2-v1/410000/790C4A52-B751-6E43-94BE-0998DB36965E.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018C-UL2018_MiniAODv2-v1/out_1154.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/JetHT/MINIAOD/UL2018_MiniAODv2-v1/410000/3113D114-7E60-E640-B8F7-C12FE9904957.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018C-UL2018_MiniAODv2-v1/out_1157.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/JetHT/MINIAOD/UL2018_MiniAODv2-v1/410000/D20DAE59-84EC-F944-B01A-E3238605E87F.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018C-UL2018_MiniAODv2-v1/out_1156.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/JetHT/MINIAOD/UL2018_MiniAODv2-v1/40000/235D0F09-B03D-C74A-8BFB-8CF9DF3E9E0C.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018C-UL2018_MiniAODv2-v1/out_1098.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/JetHT/MINIAOD/UL2018_MiniAODv2-v1/40000/F393DB20-41E2-F149-97F7-A267A8FD98CD.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018C-UL2018_MiniAODv2-v1/out_1150.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/JetHT/MINIAOD/UL2018_MiniAODv2-v1/410000/41C7A811-7055-7A4C-BE8D-85AB092271A6.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018C-UL2018_MiniAODv2-v1/out_1153.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/JetHT/MINIAOD/UL2018_MiniAODv2-v1/40000/A7A31366-177D-0242-AB24-8F4C326D4F7C.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018C-UL2018_MiniAODv2-v1/out_1152.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/JetHT/MINIAOD/UL2018_MiniAODv2-v1/280000/4DE9877D-62A6-C14C-BD36-4A754D89FD04.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018C-UL2018_MiniAODv2-v1/out_263.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/JetHT/MINIAOD/UL2018_MiniAODv2-v1/280000/8D128D29-516F-1F4F-8625-E1E16549FDC8.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018C-UL2018_MiniAODv2-v1/out_262.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/JetHT/MINIAOD/UL2018_MiniAODv2-v1/280000/116A60F8-8D11-174C-8E97-54392394A48F.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018C-UL2018_MiniAODv2-v1/out_261.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/JetHT/MINIAOD/UL2018_MiniAODv2-v1/280000/AAEFF9C0-6FC8-E248-9AD6-4A3F091725C6.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018C-UL2018_MiniAODv2-v1/out_260.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/JetHT/MINIAOD/UL2018_MiniAODv2-v1/280000/98EA31B1-E93D-1249-8184-A2BC52BCDF74.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018C-UL2018_MiniAODv2-v1/out_267.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/JetHT/MINIAOD/UL2018_MiniAODv2-v1/280000/98B8D170-59C7-894F-A41C-4289FEA2E373.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018C-UL2018_MiniAODv2-v1/out_266.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/JetHT/MINIAOD/UL2018_MiniAODv2-v1/280000/7367F8F8-56A3-AF4C-983B-6FE2A74A4E3F.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018C-UL2018_MiniAODv2-v1/out_265.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/JetHT/MINIAOD/UL2018_MiniAODv2-v1/280000/844FFB1E-5D3B-CE4C-AD19-0D2F0B93F149.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018C-UL2018_MiniAODv2-v1/out_264.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/JetHT/MINIAOD/UL2018_MiniAODv2-v1/410000/54563675-3E38-9248-AB77-C5D7A826A30B.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018C-UL2018_MiniAODv2-v1/out_1121.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/JetHT/MINIAOD/UL2018_MiniAODv2-v1/280000/A03F3287-990C-5745-9B82-680F4063577A.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018C-UL2018_MiniAODv2-v1/out_269.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/JetHT/MINIAOD/UL2018_MiniAODv2-v1/280000/DE712293-404D-F744-8DD6-A94DCE3F4065.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018C-UL2018_MiniAODv2-v1/out_268.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260000/2BF06CD6-6019-E04F-8527-6A265E57EBE2.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018C-UL2018_MiniAODv2-v1/out_478.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/JetHT/MINIAOD/UL2018_MiniAODv2-v1/50000/3967D26E-7BA6-7344-B081-19AEBC0789DB.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018C-UL2018_MiniAODv2-v1/out_59.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/JetHT/MINIAOD/UL2018_MiniAODv2-v1/50000/2E8ADB7B-1C92-5A48-A799-107BDE97D44B.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018C-UL2018_MiniAODv2-v1/out_58.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/JetHT/MINIAOD/UL2018_MiniAODv2-v1/50000/1CA7339C-3628-0F41-8BBE-CAD92F0EBECD.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018C-UL2018_MiniAODv2-v1/out_55.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/JetHT/MINIAOD/UL2018_MiniAODv2-v1/50000/A7216936-083E-5344-B511-F6085DE6FD21.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018C-UL2018_MiniAODv2-v1/out_54.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/JetHT/MINIAOD/UL2018_MiniAODv2-v1/50000/E6DC9B27-C8CA-9543-8C69-416B4F76E048.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018C-UL2018_MiniAODv2-v1/out_57.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/JetHT/MINIAOD/UL2018_MiniAODv2-v1/50000/37AEC289-CAD8-4F40-A4DB-14662BCE5478.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018C-UL2018_MiniAODv2-v1/out_56.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/JetHT/MINIAOD/UL2018_MiniAODv2-v1/50000/C0BA653A-054B-3740-811B-8DAD0C2CCAF9.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018C-UL2018_MiniAODv2-v1/out_51.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/JetHT/MINIAOD/UL2018_MiniAODv2-v1/280000/D6A6CF9C-2414-9C4E-A9FD-1ACC77229C3E.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018C-UL2018_MiniAODv2-v1/out_258.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/JetHT/MINIAOD/UL2018_MiniAODv2-v1/50000/3B45903C-596D-804B-A1AB-4CB8EC4A6B94.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018C-UL2018_MiniAODv2-v1/out_53.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/JetHT/MINIAOD/UL2018_MiniAODv2-v1/50000/03A0DD10-391F-8446-88A4-6D9165331459.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018C-UL2018_MiniAODv2-v1/out_52.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260000/091A1C55-0CB8-5440-B925-706D92C02FE1.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018C-UL2018_MiniAODv2-v1/out_537.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260000/DE56614B-44D5-6B4E-9A82-5975715515C6.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018C-UL2018_MiniAODv2-v1/out_536.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260000/E8DC4965-9629-5D4F-A87F-322538617725.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018C-UL2018_MiniAODv2-v1/out_535.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/JetHT/MINIAOD/UL2018_MiniAODv2-v1/50000/6FBE1B97-269F-324F-8AA0-D357D7B0A6C9.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018C-UL2018_MiniAODv2-v1/out_63.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260000/2D9A4100-5314-4448-B6D0-20678FF10D10.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018C-UL2018_MiniAODv2-v1/out_533.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260000/EA773BF7-FD42-3C47-9471-AABD7232A34C.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018C-UL2018_MiniAODv2-v1/out_532.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260000/F137ABA5-ED33-4141-9640-24C38BC407AE.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018C-UL2018_MiniAODv2-v1/out_531.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260000/62C5CA3E-724A-014F-A7A2-F822080A638B.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018C-UL2018_MiniAODv2-v1/out_530.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/JetHT/MINIAOD/UL2018_MiniAODv2-v1/50000/A7EAED0F-8A20-1D4A-A05B-FC3FE6065588.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018C-UL2018_MiniAODv2-v1/out_598.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/JetHT/MINIAOD/UL2018_MiniAODv2-v1/40000/C58F5BD3-011A-C648-994F-30AFF2573AD7.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018C-UL2018_MiniAODv2-v1/out_152.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260000/DE0CE1AA-3838-DF41-B1EF-55E308AECA39.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018C-UL2018_MiniAODv2-v1/out_539.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260000/7913A216-DF4B-E446-969E-AFFD0791007C.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018C-UL2018_MiniAODv2-v1/out_538.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/JetHT/MINIAOD/UL2018_MiniAODv2-v1/50000/C5FD33E6-5C0B-484C-9719-09EA0B7C7E60.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018C-UL2018_MiniAODv2-v1/out_775.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/JetHT/MINIAOD/UL2018_MiniAODv2-v1/50000/EF9FDEE4-21B7-0840-8BF3-4E346AD5F5FC.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018C-UL2018_MiniAODv2-v1/out_988.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/JetHT/MINIAOD/UL2018_MiniAODv2-v1/50000/7864A96E-DFCE-844A-ABAF-CA943F9ADABF.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018C-UL2018_MiniAODv2-v1/out_989.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/JetHT/MINIAOD/UL2018_MiniAODv2-v1/410000/F3854897-EFCC-DC4D-936C-5E4AA472AA30.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018C-UL2018_MiniAODv2-v1/out_1171.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/JetHT/MINIAOD/UL2018_MiniAODv2-v1/50000/61FEB40A-F435-5643-A9F5-BAF35EAB47DB.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018C-UL2018_MiniAODv2-v1/out_774.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/JetHT/MINIAOD/UL2018_MiniAODv2-v1/50000/C689DC34-B3E2-E049-9B2D-73EE11C12961.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018C-UL2018_MiniAODv2-v1/out_982.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/JetHT/MINIAOD/UL2018_MiniAODv2-v1/50000/2D903D72-3DF6-654F-9085-0A20A4E909CC.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018C-UL2018_MiniAODv2-v1/out_983.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/JetHT/MINIAOD/UL2018_MiniAODv2-v1/50000/E61462DD-8A51-9248-81AD-9EFB4136BB80.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018C-UL2018_MiniAODv2-v1/out_980.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/JetHT/MINIAOD/UL2018_MiniAODv2-v1/50000/65F92C75-26C6-A746-9645-43E0619F92BC.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018C-UL2018_MiniAODv2-v1/out_981.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/JetHT/MINIAOD/UL2018_MiniAODv2-v1/50000/D7451074-75DA-9C48-BFF6-DAFD387A386B.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018C-UL2018_MiniAODv2-v1/out_986.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/JetHT/MINIAOD/UL2018_MiniAODv2-v1/50000/22D1A877-1DED-054E-B42C-19E9C7369076.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018C-UL2018_MiniAODv2-v1/out_987.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/JetHT/MINIAOD/UL2018_MiniAODv2-v1/50000/19C66F5E-A1CA-F448-8E14-FD208431C89A.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018C-UL2018_MiniAODv2-v1/out_984.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/JetHT/MINIAOD/UL2018_MiniAODv2-v1/50000/2771BCC4-77EC-2D4A-BCCA-A218A1C00222.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018C-UL2018_MiniAODv2-v1/out_50.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/JetHT/MINIAOD/UL2018_MiniAODv2-v1/50000/F9A8E09C-6ADA-B945-B4B3-D5332AABF0CC.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018C-UL2018_MiniAODv2-v1/out_115.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/JetHT/MINIAOD/UL2018_MiniAODv2-v1/280000/A8BDF302-2974-9C4A-AFEF-7BD6013AE927.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018C-UL2018_MiniAODv2-v1/out_252.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/JetHT/MINIAOD/UL2018_MiniAODv2-v1/50000/F59E988E-6ED3-134D-92EF-137E91E4AE65.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018C-UL2018_MiniAODv2-v1/out_117.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/JetHT/MINIAOD/UL2018_MiniAODv2-v1/50000/9FC48EE0-E305-A84B-84E8-69C80F7F89BD.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018C-UL2018_MiniAODv2-v1/out_776.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/JetHT/MINIAOD/UL2018_MiniAODv2-v1/50000/BBC5CE90-DF33-854E-BD72-1739603CC8BE.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018C-UL2018_MiniAODv2-v1/out_111.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/JetHT/MINIAOD/UL2018_MiniAODv2-v1/50000/6CDD32F3-4786-6743-9139-2F277DFC85B4.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018C-UL2018_MiniAODv2-v1/out_110.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/JetHT/MINIAOD/UL2018_MiniAODv2-v1/50000/2DE180C8-78FF-8C4E-9138-61E8BACEED09.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018C-UL2018_MiniAODv2-v1/out_113.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/JetHT/MINIAOD/UL2018_MiniAODv2-v1/50000/7BDD7D98-3967-EA45-91C7-4BB17F9A2355.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018C-UL2018_MiniAODv2-v1/out_69.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/JetHT/MINIAOD/UL2018_MiniAODv2-v1/50000/2CE2A297-332B-F744-8946-F0B1A664ECD0.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018C-UL2018_MiniAODv2-v1/out_677.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/JetHT/MINIAOD/UL2018_MiniAODv2-v1/50000/E656248A-B057-8441-A4E9-13AD58CCD32D.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018C-UL2018_MiniAODv2-v1/out_771.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/JetHT/MINIAOD/UL2018_MiniAODv2-v1/50000/AF1DCB90-F938-A34D-AC6B-2C4620D5ADB8.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018C-UL2018_MiniAODv2-v1/out_5.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/JetHT/MINIAOD/UL2018_MiniAODv2-v1/50000/B787C74A-CF60-5A4D-984B-EED86277D427.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018C-UL2018_MiniAODv2-v1/out_119.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/JetHT/MINIAOD/UL2018_MiniAODv2-v1/50000/AAB66B55-1723-0C42-8EA3-CC1654F6E7F1.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018C-UL2018_MiniAODv2-v1/out_118.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/JetHT/MINIAOD/UL2018_MiniAODv2-v1/50000/4910FD7C-DA8F-2641-AB56-B1C4AC1E28C1.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018C-UL2018_MiniAODv2-v1/out_770.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/JetHT/MINIAOD/UL2018_MiniAODv2-v1/50000/5A5FC250-D826-7549-BA4C-67BE5BC2D9C0.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018C-UL2018_MiniAODv2-v1/out_773.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/JetHT/MINIAOD/UL2018_MiniAODv2-v1/50000/56AD8F81-E33B-BB46-8DA4-68C45BCAE23B.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018C-UL2018_MiniAODv2-v1/out_772.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/JetHT/MINIAOD/UL2018_MiniAODv2-v1/50000/76EA54E2-27C3-064F-B4A2-D130DE825F98.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018C-UL2018_MiniAODv2-v1/out_1001.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/JetHT/MINIAOD/UL2018_MiniAODv2-v1/50000/D95A5EBD-6E71-1042-A709-B87BA7B95C29.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018C-UL2018_MiniAODv2-v1/out_429.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/JetHT/MINIAOD/UL2018_MiniAODv2-v1/50000/F10621CC-5379-6844-B8A0-D662330C0544.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018C-UL2018_MiniAODv2-v1/out_428.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260000/2590B435-963F-0A45-9371-C1D3E65E627D.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018C-UL2018_MiniAODv2-v1/out_534.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/JetHT/MINIAOD/UL2018_MiniAODv2-v1/40000/EE3A55E1-A4BF-8244-B739-FFBD22642550.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018C-UL2018_MiniAODv2-v1/out_919.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/JetHT/MINIAOD/UL2018_MiniAODv2-v1/40000/16FED854-E5FA-8A47-8573-9AA707ACFEFA.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018C-UL2018_MiniAODv2-v1/out_918.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/JetHT/MINIAOD/UL2018_MiniAODv2-v1/40000/92D06433-61DA-7F45-BFA2-6531E721FAF1.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018C-UL2018_MiniAODv2-v1/out_915.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/JetHT/MINIAOD/UL2018_MiniAODv2-v1/50000/419BED4D-A97F-D24E-BB56-2CE336042B32.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018C-UL2018_MiniAODv2-v1/out_914.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/JetHT/MINIAOD/UL2018_MiniAODv2-v1/40000/25F49AFA-0D5A-0148-9974-668A10189297.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018C-UL2018_MiniAODv2-v1/out_917.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/JetHT/MINIAOD/UL2018_MiniAODv2-v1/40000/27F4EA17-6C09-A94D-A748-22F0BC65D87F.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018C-UL2018_MiniAODv2-v1/out_916.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/JetHT/MINIAOD/UL2018_MiniAODv2-v1/50000/FD4FC69D-84B9-AA43-BCFE-2D35FA652318.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018C-UL2018_MiniAODv2-v1/out_911.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/JetHT/MINIAOD/UL2018_MiniAODv2-v1/280000/86B4E451-2B53-BD4A-871A-BAAAB120BBF4.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018C-UL2018_MiniAODv2-v1/out_910.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/JetHT/MINIAOD/UL2018_MiniAODv2-v1/50000/D15A04F3-8FC5-3641-954D-AF7AC556E993.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018C-UL2018_MiniAODv2-v1/out_913.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/JetHT/MINIAOD/UL2018_MiniAODv2-v1/50000/E89ABF99-9E2C-4E40-8840-D65C33B4151D.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018C-UL2018_MiniAODv2-v1/out_912.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/JetHT/MINIAOD/UL2018_MiniAODv2-v1/40000/6E45C035-95B4-7748-8E9D-80D8FDC5C1A5.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018C-UL2018_MiniAODv2-v1/out_308.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/JetHT/MINIAOD/UL2018_MiniAODv2-v1/40000/34BB8941-A95D-D14E-ADF3-6D67224FC67A.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018C-UL2018_MiniAODv2-v1/out_309.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/JetHT/MINIAOD/UL2018_MiniAODv2-v1/280000/DC6D3F37-7C26-9D47-A1D2-0BDB05A660AB.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018C-UL2018_MiniAODv2-v1/out_855.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/JetHT/MINIAOD/UL2018_MiniAODv2-v1/40000/B4ADC8B6-92AA-A34A-8B46-8B0FBC961663.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018C-UL2018_MiniAODv2-v1/out_300.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/JetHT/MINIAOD/UL2018_MiniAODv2-v1/40000/57FC2223-6C41-B64C-9C0D-46ADF6E7A805.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018C-UL2018_MiniAODv2-v1/out_301.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/JetHT/MINIAOD/UL2018_MiniAODv2-v1/40000/5240BE42-9B4D-754F-9C08-BC243DF20188.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018C-UL2018_MiniAODv2-v1/out_302.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/JetHT/MINIAOD/UL2018_MiniAODv2-v1/40000/0770D5C5-9EB9-6D4F-9DD0-7A8CF88A7602.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018C-UL2018_MiniAODv2-v1/out_303.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/JetHT/MINIAOD/UL2018_MiniAODv2-v1/40000/53C7CA9C-436B-F643-AD49-63FCFD88F1E6.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018C-UL2018_MiniAODv2-v1/out_304.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/JetHT/MINIAOD/UL2018_MiniAODv2-v1/40000/444B925C-BC7B-0448-8C43-40D9C616DA5A.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018C-UL2018_MiniAODv2-v1/out_305.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/JetHT/MINIAOD/UL2018_MiniAODv2-v1/40000/5316B2E3-E5BF-CF45-8B74-44A92713CBAD.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018C-UL2018_MiniAODv2-v1/out_306.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/JetHT/MINIAOD/UL2018_MiniAODv2-v1/40000/0B6212F3-E7F3-8A45-AC10-ED1D9171D770.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018C-UL2018_MiniAODv2-v1/out_307.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/JetHT/MINIAOD/UL2018_MiniAODv2-v1/280000/4C32D602-F7A9-8D40-A2AB-7994F533F05B.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018C-UL2018_MiniAODv2-v1/out_895.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/JetHT/MINIAOD/UL2018_MiniAODv2-v1/40000/014F85B4-23FE-954E-A40B-F139C287C050.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018C-UL2018_MiniAODv2-v1/out_825.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/JetHT/MINIAOD/UL2018_MiniAODv2-v1/50000/34437437-CE96-B445-A1B7-4B8331DAECAD.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018C-UL2018_MiniAODv2-v1/out_665.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/JetHT/MINIAOD/UL2018_MiniAODv2-v1/40000/38508F9B-F21C-A940-8547-8E8580EDF10D.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018C-UL2018_MiniAODv2-v1/out_824.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/JetHT/MINIAOD/UL2018_MiniAODv2-v1/50000/56A881F5-16EB-F543-A400-E2E9D783B724.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018C-UL2018_MiniAODv2-v1/out_649.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/JetHT/MINIAOD/UL2018_MiniAODv2-v1/280000/AA69E77A-A5B8-F940-B2A1-A01B75A0CD05.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018C-UL2018_MiniAODv2-v1/out_847.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/JetHT/MINIAOD/UL2018_MiniAODv2-v1/280000/89B75BDA-7A04-1242-BA3E-B4EC4DC82118.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018C-UL2018_MiniAODv2-v1/out_846.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/JetHT/MINIAOD/UL2018_MiniAODv2-v1/280000/1BCD0A31-5DA4-3740-80C6-24E81F852912.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018C-UL2018_MiniAODv2-v1/out_845.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/JetHT/MINIAOD/UL2018_MiniAODv2-v1/280000/BDDB63F8-5904-034A-B609-A2E3994C3521.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018C-UL2018_MiniAODv2-v1/out_844.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/JetHT/MINIAOD/UL2018_MiniAODv2-v1/280000/1EBB7A90-FBE3-9C45-9662-A7A3BD558DD4.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018C-UL2018_MiniAODv2-v1/out_843.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/JetHT/MINIAOD/UL2018_MiniAODv2-v1/280000/4F43F42F-4D24-864B-B560-934D9867E08A.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018C-UL2018_MiniAODv2-v1/out_842.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/JetHT/MINIAOD/UL2018_MiniAODv2-v1/280000/0BCAA5EB-BF50-F44E-BF38-189C699D1AD9.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018C-UL2018_MiniAODv2-v1/out_841.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/JetHT/MINIAOD/UL2018_MiniAODv2-v1/40000/573BB318-813F-9B42-BBC3-A6638B40AA43.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018C-UL2018_MiniAODv2-v1/out_840.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/JetHT/MINIAOD/UL2018_MiniAODv2-v1/40000/9608F365-D28C-5F42-B916-AB0DD686C579.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018C-UL2018_MiniAODv2-v1/out_1151.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/JetHT/MINIAOD/UL2018_MiniAODv2-v1/40000/255300D8-4EED-8D4A-887B-6619BA850731.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018C-UL2018_MiniAODv2-v1/out_821.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/JetHT/MINIAOD/UL2018_MiniAODv2-v1/280000/637A8786-0DB0-A244-99DE-368EF94C2C1B.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018C-UL2018_MiniAODv2-v1/out_853.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/JetHT/MINIAOD/UL2018_MiniAODv2-v1/410000/6D9F863F-F402-0648-9AF5-A5B08C09628A.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018C-UL2018_MiniAODv2-v1/out_1126.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/JetHT/MINIAOD/UL2018_MiniAODv2-v1/280000/22894249-571D-1C42-8CE2-B06520AB0ED8.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018C-UL2018_MiniAODv2-v1/out_848.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/JetHT/MINIAOD/UL2018_MiniAODv2-v1/40000/EBF2199B-3368-8F47-A408-B3749DD134A4.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018C-UL2018_MiniAODv2-v1/out_823.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/JetHT/MINIAOD/UL2018_MiniAODv2-v1/40000/1646055C-918E-6548-92F0-400D728D3571.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018C-UL2018_MiniAODv2-v1/out_822.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/JetHT/MINIAOD/UL2018_MiniAODv2-v1/410000/7ED0E66B-B7E5-6048-B44D-56D626A9292F.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018C-UL2018_MiniAODv2-v1/out_1127.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/JetHT/MINIAOD/UL2018_MiniAODv2-v1/40000/8ED665AB-579F-9440-897A-FCDABB571167.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018C-UL2018_MiniAODv2-v1/out_240.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260000/4E9E3B99-A8BD-804D-8E7F-F450C7716B7C.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018C-UL2018_MiniAODv2-v1/out_485.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/JetHT/MINIAOD/UL2018_MiniAODv2-v1/50000/F8122FC2-9A43-1348-9668-B6B348D56783.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018C-UL2018_MiniAODv2-v1/out_390.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/JetHT/MINIAOD/UL2018_MiniAODv2-v1/50000/19C504D8-162A-9B4C-8E7A-20AC211D42DB.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018C-UL2018_MiniAODv2-v1/out_1038.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/JetHT/MINIAOD/UL2018_MiniAODv2-v1/50000/4F1F3717-1BCE-8E4C-9DCC-9DA305009299.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018C-UL2018_MiniAODv2-v1/out_732.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/JetHT/MINIAOD/UL2018_MiniAODv2-v1/40000/FFA5528A-00C3-D242-AAED-FDC2DFCAF196.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018C-UL2018_MiniAODv2-v1/out_162.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/JetHT/MINIAOD/UL2018_MiniAODv2-v1/410000/44660258-29A8-8B46-8605-160DFB5BCEEB.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018C-UL2018_MiniAODv2-v1/out_1166.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/JetHT/MINIAOD/UL2018_MiniAODv2-v1/50000/272EDBD1-95D6-494F-B2C0-DC562714C8A7.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018C-UL2018_MiniAODv2-v1/out_391.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/JetHT/MINIAOD/UL2018_MiniAODv2-v1/50000/B2805545-1009-AF4E-AC9C-AA3843708CF1.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018C-UL2018_MiniAODv2-v1/out_753.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/JetHT/MINIAOD/UL2018_MiniAODv2-v1/50000/07A95891-52AE-9E4F-B670-F7BFC3FEAAAE.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018C-UL2018_MiniAODv2-v1/out_752.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/JetHT/MINIAOD/UL2018_MiniAODv2-v1/50000/6115C8C6-A4B9-5744-8245-DD9829EB2407.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018C-UL2018_MiniAODv2-v1/out_751.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/JetHT/MINIAOD/UL2018_MiniAODv2-v1/50000/D112FA66-49EC-4743-8E00-B9DAB05E0CE2.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018C-UL2018_MiniAODv2-v1/out_750.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/JetHT/MINIAOD/UL2018_MiniAODv2-v1/50000/EFA2354A-6212-7746-960D-0B03654532FF.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018C-UL2018_MiniAODv2-v1/out_757.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/JetHT/MINIAOD/UL2018_MiniAODv2-v1/50000/8CAA9FDB-FA64-A246-8352-17744C78BB4E.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018C-UL2018_MiniAODv2-v1/out_756.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/JetHT/MINIAOD/UL2018_MiniAODv2-v1/50000/15A0E01A-C10D-BE40-A5E2-D5B314DED6E7.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018C-UL2018_MiniAODv2-v1/out_755.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/JetHT/MINIAOD/UL2018_MiniAODv2-v1/50000/149FCF9B-A0B7-5240-964F-62D0B17237E2.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018C-UL2018_MiniAODv2-v1/out_754.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/JetHT/MINIAOD/UL2018_MiniAODv2-v1/50000/F2FB9935-AF33-E34F-B6E0-9A42523D06CE.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018C-UL2018_MiniAODv2-v1/out_560.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/JetHT/MINIAOD/UL2018_MiniAODv2-v1/50000/BF288375-D640-6B46-9C98-6C27C391F8B5.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018C-UL2018_MiniAODv2-v1/out_561.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/JetHT/MINIAOD/UL2018_MiniAODv2-v1/50000/DC28330F-9B44-C048-BF80-97E71A36295E.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018C-UL2018_MiniAODv2-v1/out_759.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/JetHT/MINIAOD/UL2018_MiniAODv2-v1/50000/C6DD9D55-9AEE-544D-BA70-75EFD63ED2D7.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018C-UL2018_MiniAODv2-v1/out_758.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/JetHT/MINIAOD/UL2018_MiniAODv2-v1/50000/57AAF0AD-8A64-714A-8634-BF0AD49826B5.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018C-UL2018_MiniAODv2-v1/out_564.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/JetHT/MINIAOD/UL2018_MiniAODv2-v1/50000/C76BFCC0-9017-0647-A9F3-5ACD7B348E9F.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018C-UL2018_MiniAODv2-v1/out_565.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/JetHT/MINIAOD/UL2018_MiniAODv2-v1/50000/7CC5580E-7B5B-1340-92D6-C34CAD87E7A2.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018C-UL2018_MiniAODv2-v1/out_566.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/JetHT/MINIAOD/UL2018_MiniAODv2-v1/50000/C4705558-BC0C-1046-9922-2BDF2408A119.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018C-UL2018_MiniAODv2-v1/out_701.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260000/FFE35F7A-1786-BE4F-8AF4-66DEA58012F3.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018C-UL2018_MiniAODv2-v1/out_506.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/JetHT/MINIAOD/UL2018_MiniAODv2-v1/40000/0B86690D-D806-C240-95D9-857836795663.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018C-UL2018_MiniAODv2-v1/out_229.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260000/39488C8F-EB71-E849-8048-5131FE06806D.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018C-UL2018_MiniAODv2-v1/out_507.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/JetHT/MINIAOD/UL2018_MiniAODv2-v1/40000/7FF67839-6DF6-914C-A5DF-EA27000F38E8.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018C-UL2018_MiniAODv2-v1/out_227.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/JetHT/MINIAOD/UL2018_MiniAODv2-v1/40000/1877EE0B-E5D2-124E-B1C5-4BC49F797A1C.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018C-UL2018_MiniAODv2-v1/out_226.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/JetHT/MINIAOD/UL2018_MiniAODv2-v1/40000/4B7EF2DB-3361-9E47-9C26-9DD6F654954F.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018C-UL2018_MiniAODv2-v1/out_225.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/JetHT/MINIAOD/UL2018_MiniAODv2-v1/50000/FBEE186C-1D8C-D148-8B4F-49AB2DB9A91D.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018C-UL2018_MiniAODv2-v1/out_985.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/JetHT/MINIAOD/UL2018_MiniAODv2-v1/40000/C6B58372-1F2E-0F4D-B130-B8C3A4DAB2B8.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018C-UL2018_MiniAODv2-v1/out_223.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260000/51738D5C-1794-DF4D-85C0-22421F1781CC.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018C-UL2018_MiniAODv2-v1/out_504.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/JetHT/MINIAOD/UL2018_MiniAODv2-v1/40000/5A0AED1B-6550-AD4F-8B2D-000F11B89631.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018C-UL2018_MiniAODv2-v1/out_221.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/JetHT/MINIAOD/UL2018_MiniAODv2-v1/40000/A3F69BF1-D8A3-504A-BEB7-E6AE76DD6559.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018C-UL2018_MiniAODv2-v1/out_220.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/JetHT/MINIAOD/UL2018_MiniAODv2-v1/50000/4B6D0510-8D46-AA4B-9AAC-600DEDF2FDA7.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018C-UL2018_MiniAODv2-v1/out_1025.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/JetHT/MINIAOD/UL2018_MiniAODv2-v1/50000/75B37C3F-E2DC-804E-8862-124350822613.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018C-UL2018_MiniAODv2-v1/out_1024.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/JetHT/MINIAOD/UL2018_MiniAODv2-v1/50000/C26A00CE-76FC-7F4A-A4CF-E61B47D2E89A.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018C-UL2018_MiniAODv2-v1/out_1027.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/JetHT/MINIAOD/UL2018_MiniAODv2-v1/50000/E99AFB3C-5BCF-C342-842B-117BA3F56B64.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018C-UL2018_MiniAODv2-v1/out_1031.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/JetHT/MINIAOD/UL2018_MiniAODv2-v1/50000/F24BE677-EAFD-9142-A09A-C2B012534FDA.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018C-UL2018_MiniAODv2-v1/out_1021.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/JetHT/MINIAOD/UL2018_MiniAODv2-v1/50000/E1471057-DD63-8343-9ECC-A7FCB82DF540.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018C-UL2018_MiniAODv2-v1/out_1020.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/JetHT/MINIAOD/UL2018_MiniAODv2-v1/50000/C1EA7EA7-B88F-B440-A28F-BA85C2111B96.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018C-UL2018_MiniAODv2-v1/out_1023.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/JetHT/MINIAOD/UL2018_MiniAODv2-v1/50000/CF9446C7-6A8C-3149-9F7D-26A4515024F2.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018C-UL2018_MiniAODv2-v1/out_114.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/JetHT/MINIAOD/UL2018_MiniAODv2-v1/230000/D428CE58-A4C0-CE4B-83D5-CADB1C46BFF4.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018C-UL2018_MiniAODv2-v1/out_1186.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/JetHT/MINIAOD/UL2018_MiniAODv2-v1/140000/160530F1-69F9-8242-BABE-F18AF0A45A4D.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018C-UL2018_MiniAODv2-v1/out_1187.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/JetHT/MINIAOD/UL2018_MiniAODv2-v1/00000/CE9AAE3E-F928-884D-8FB3-9A6A347D7A28.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018C-UL2018_MiniAODv2-v1/out_1184.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/JetHT/MINIAOD/UL2018_MiniAODv2-v1/30000/DE740B5B-264D-6B4C-8E30-1DD922A051EF.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018C-UL2018_MiniAODv2-v1/out_1185.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/JetHT/MINIAOD/UL2018_MiniAODv2-v1/00000/9F030B27-DBB0-EE46-A8FB-64FE0C417EE9.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018C-UL2018_MiniAODv2-v1/out_1182.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/JetHT/MINIAOD/UL2018_MiniAODv2-v1/50000/3DE07DD5-DE12-E64C-BE29-6C3EFD219829.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018C-UL2018_MiniAODv2-v1/out_1028.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/JetHT/MINIAOD/UL2018_MiniAODv2-v1/50000/55F86FBD-0235-BC4E-886C-B423A5A19160.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018C-UL2018_MiniAODv2-v1/out_1180.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/JetHT/MINIAOD/UL2018_MiniAODv2-v1/50000/EC07F1C5-200F-BC45-9449-0B8C0D2308EA.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018C-UL2018_MiniAODv2-v1/out_1181.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/JetHT/MINIAOD/UL2018_MiniAODv2-v1/50000/E0661A0D-35ED-184E-8409-8E57BDF5D01B.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018C-UL2018_MiniAODv2-v1/out_726.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/JetHT/MINIAOD/UL2018_MiniAODv2-v1/50000/1AEC065B-43E9-A34F-A39C-A9C5A80AC1A1.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018C-UL2018_MiniAODv2-v1/out_727.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/JetHT/MINIAOD/UL2018_MiniAODv2-v1/50000/E3896FF0-8C0B-9243-BB77-F5893E188393.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018C-UL2018_MiniAODv2-v1/out_724.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/JetHT/MINIAOD/UL2018_MiniAODv2-v1/50000/F64717A5-C3D2-F847-80DD-B15F33A8658E.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018C-UL2018_MiniAODv2-v1/out_89.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/JetHT/MINIAOD/UL2018_MiniAODv2-v1/50000/EC1B0544-80BF-B149-849C-30994C739B27.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018C-UL2018_MiniAODv2-v1/out_568.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/JetHT/MINIAOD/UL2018_MiniAODv2-v1/50000/BCE7753B-C0C4-1D4E-AB6E-A70580A65F6F.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018C-UL2018_MiniAODv2-v1/out_1034.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/JetHT/MINIAOD/UL2018_MiniAODv2-v1/50000/4D714D37-C76A-704C-93F2-1E11FA3B8E65.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018C-UL2018_MiniAODv2-v1/out_720.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/JetHT/MINIAOD/UL2018_MiniAODv2-v1/50000/E4D6EBEF-92B6-AC49-837A-98BC9E1C305A.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018C-UL2018_MiniAODv2-v1/out_721.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/JetHT/MINIAOD/UL2018_MiniAODv2-v1/410000/DE1B9145-3DFA-3645-A149-165D63B98C82.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018C-UL2018_MiniAODv2-v1/out_1160.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/JetHT/MINIAOD/UL2018_MiniAODv2-v1/50000/EA434EA4-ACE0-0445-9CEA-E384801D615B.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018C-UL2018_MiniAODv2-v1/out_1035.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/JetHT/MINIAOD/UL2018_MiniAODv2-v1/410000/6A735646-9B84-7A42-8418-D491D6FE623D.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018C-UL2018_MiniAODv2-v1/out_1167.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/JetHT/MINIAOD/UL2018_MiniAODv2-v1/50000/1A41E38A-8C4A-1C4C-B008-AFBC328F7155.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018C-UL2018_MiniAODv2-v1/out_728.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/JetHT/MINIAOD/UL2018_MiniAODv2-v1/50000/008F1FAF-CB98-AF4E-A806-8FCFFC961856.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018C-UL2018_MiniAODv2-v1/out_729.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/JetHT/MINIAOD/UL2018_MiniAODv2-v1/50000/BCE50BD6-F351-9E44-99CE-79C5D052792B.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018C-UL2018_MiniAODv2-v1/out_605.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/JetHT/MINIAOD/UL2018_MiniAODv2-v1/50000/88A061F7-13EB-6B4D-9B6A-5D8E4DE01A3D.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018C-UL2018_MiniAODv2-v1/out_604.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/JetHT/MINIAOD/UL2018_MiniAODv2-v1/50000/8FE9942E-9BAF-C04C-B0B8-231806B80226.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018C-UL2018_MiniAODv2-v1/out_607.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/JetHT/MINIAOD/UL2018_MiniAODv2-v1/50000/06789617-5397-A14F-941D-E7F08CA76354.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018C-UL2018_MiniAODv2-v1/out_606.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/JetHT/MINIAOD/UL2018_MiniAODv2-v1/50000/ABA2F059-C8F1-6B40-8614-B2A16DF8A8B5.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018C-UL2018_MiniAODv2-v1/out_601.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/JetHT/MINIAOD/UL2018_MiniAODv2-v1/50000/875B0996-C021-164D-80CD-5A62FC7E6C40.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018C-UL2018_MiniAODv2-v1/out_600.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/JetHT/MINIAOD/UL2018_MiniAODv2-v1/410000/FD2EEC8B-3282-0B49-B40B-C6D3E3F7EACA.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018C-UL2018_MiniAODv2-v1/out_1162.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/JetHT/MINIAOD/UL2018_MiniAODv2-v1/50000/54EA5245-FF9B-1645-A57A-2590D36156D1.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018C-UL2018_MiniAODv2-v1/out_602.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/JetHT/MINIAOD/UL2018_MiniAODv2-v1/40000/6D5FB24B-40D3-9C4C-B81A-3DDF97054ADE.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018C-UL2018_MiniAODv2-v1/out_159.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/JetHT/MINIAOD/UL2018_MiniAODv2-v1/40000/114E1163-3F15-1049-9B9B-BDECFA0ED157.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018C-UL2018_MiniAODv2-v1/out_158.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/JetHT/MINIAOD/UL2018_MiniAODv2-v1/50000/47C89594-A415-D949-843A-940B4E7CB501.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018C-UL2018_MiniAODv2-v1/out_112.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/JetHT/MINIAOD/UL2018_MiniAODv2-v1/410000/5A84ED50-3991-EA48-A68D-088217CBC599.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018C-UL2018_MiniAODv2-v1/out_1168.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/JetHT/MINIAOD/UL2018_MiniAODv2-v1/410000/7A733183-E452-E448-9E15-03BFE220D9D4.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018C-UL2018_MiniAODv2-v1/out_1169.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/JetHT/MINIAOD/UL2018_MiniAODv2-v1/50000/C73D906F-866A-A64C-9918-A49EB68317DF.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018C-UL2018_MiniAODv2-v1/out_976.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/JetHT/MINIAOD/UL2018_MiniAODv2-v1/410000/2720D45B-EDC4-B244-9661-A9BD0A0AE927.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018C-UL2018_MiniAODv2-v1/out_1161.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/JetHT/MINIAOD/UL2018_MiniAODv2-v1/50000/6646BFB5-509C-5A4B-BA27-CFE6D6A5A43F.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018C-UL2018_MiniAODv2-v1/out_634.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/JetHT/MINIAOD/UL2018_MiniAODv2-v1/40000/6EAFD2B0-C1C2-F24C-B114-D22760EE3AD8.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018C-UL2018_MiniAODv2-v1/out_935.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/JetHT/MINIAOD/UL2018_MiniAODv2-v1/50000/867565D1-3CB7-5E42-A3AD-8ABB6C3FF77C.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018C-UL2018_MiniAODv2-v1/out_399.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/JetHT/MINIAOD/UL2018_MiniAODv2-v1/50000/503F47EE-DD6F-5F4A-BDFE-116FFCE1382F.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018C-UL2018_MiniAODv2-v1/out_635.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/JetHT/MINIAOD/UL2018_MiniAODv2-v1/50000/43557588-ED30-0B45-8330-34996744E8A7.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018C-UL2018_MiniAODv2-v1/out_959.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/JetHT/MINIAOD/UL2018_MiniAODv2-v1/50000/AFBBD463-761F-CE42-B49B-BB652CDCCF3C.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018C-UL2018_MiniAODv2-v1/out_958.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/JetHT/MINIAOD/UL2018_MiniAODv2-v1/50000/B0193B58-F5E5-2B42-AAFF-91D623878CB3.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018C-UL2018_MiniAODv2-v1/out_398.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/JetHT/MINIAOD/UL2018_MiniAODv2-v1/50000/D8D2F537-2294-2540-AF25-111F17960663.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018C-UL2018_MiniAODv2-v1/out_48.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/JetHT/MINIAOD/UL2018_MiniAODv2-v1/50000/7F08197B-BC5C-7B46-A3E6-B5095FA30643.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018C-UL2018_MiniAODv2-v1/out_49.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/JetHT/MINIAOD/UL2018_MiniAODv2-v1/40000/CD3FC698-15A8-0040-A0C5-3A780754AF4A.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018C-UL2018_MiniAODv2-v1/out_951.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/JetHT/MINIAOD/UL2018_MiniAODv2-v1/40000/07C5EE4B-21E0-C642-9203-5AE2FBAFC774.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018C-UL2018_MiniAODv2-v1/out_950.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/JetHT/MINIAOD/UL2018_MiniAODv2-v1/40000/8F431396-6E74-5A4E-903A-CABB38D74CBA.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018C-UL2018_MiniAODv2-v1/out_953.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/JetHT/MINIAOD/UL2018_MiniAODv2-v1/40000/E3A71F55-E3FC-924A-836A-CDE5EE67624D.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018C-UL2018_MiniAODv2-v1/out_952.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/JetHT/MINIAOD/UL2018_MiniAODv2-v1/50000/2558E2A7-7AF5-A343-8BA0-E752CF31FB85.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018C-UL2018_MiniAODv2-v1/out_42.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/JetHT/MINIAOD/UL2018_MiniAODv2-v1/40000/A4F29F9B-3C54-5245-BDD1-4AEB3CC0060E.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018C-UL2018_MiniAODv2-v1/out_954.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/JetHT/MINIAOD/UL2018_MiniAODv2-v1/40000/A1D02A97-D411-2145-9927-3F2BACB0BB5E.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018C-UL2018_MiniAODv2-v1/out_957.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/JetHT/MINIAOD/UL2018_MiniAODv2-v1/40000/2C0A6479-9592-FD4F-83FB-F70361ECCE6C.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018C-UL2018_MiniAODv2-v1/out_956.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/JetHT/MINIAOD/UL2018_MiniAODv2-v1/40000/E9D0AA4B-49A5-B44E-BE2B-2923D4ABFD6A.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018C-UL2018_MiniAODv2-v1/out_1111.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/JetHT/MINIAOD/UL2018_MiniAODv2-v1/40000/F82E38ED-480C-074F-9545-9B7924FBC27D.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018C-UL2018_MiniAODv2-v1/out_1110.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/JetHT/MINIAOD/UL2018_MiniAODv2-v1/410000/733745F6-DF5D-1949-A48A-03B397434B90.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018C-UL2018_MiniAODv2-v1/out_1113.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/JetHT/MINIAOD/UL2018_MiniAODv2-v1/410000/D36BE1E3-6071-D84A-95E1-CB8EBAACB9BE.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018C-UL2018_MiniAODv2-v1/out_1112.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/JetHT/MINIAOD/UL2018_MiniAODv2-v1/410000/4C9A65A0-D2AF-7142-9327-BB1975954F5A.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018C-UL2018_MiniAODv2-v1/out_1115.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/JetHT/MINIAOD/UL2018_MiniAODv2-v1/410000/59BA3FEB-C4C2-7942-971F-71CB4C169D61.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018C-UL2018_MiniAODv2-v1/out_1114.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/JetHT/MINIAOD/UL2018_MiniAODv2-v1/410000/E424BD99-05E1-C24C-93DF-B85D0B8496D8.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018C-UL2018_MiniAODv2-v1/out_1117.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/JetHT/MINIAOD/UL2018_MiniAODv2-v1/410000/9277D4D6-7A07-1C47-83EE-D5BB807F3D03.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018C-UL2018_MiniAODv2-v1/out_1116.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/JetHT/MINIAOD/UL2018_MiniAODv2-v1/50000/2CE6ABD8-395B-9849-8EBC-24B5C1FE6007.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018C-UL2018_MiniAODv2-v1/out_620.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/JetHT/MINIAOD/UL2018_MiniAODv2-v1/50000/48E954CB-BFA6-5C41-87A4-2085E9E15466.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018C-UL2018_MiniAODv2-v1/out_608.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260000/0F64D56B-383D-3246-BDC1-4538E3A9BC32.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018C-UL2018_MiniAODv2-v1/out_467.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260000/D182084C-8580-2245-97D2-84270069C5D5.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018C-UL2018_MiniAODv2-v1/out_460.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/JetHT/MINIAOD/UL2018_MiniAODv2-v1/410000/732045EA-9B78-C34C-A48B-29E8D60EE5C0.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018C-UL2018_MiniAODv2-v1/out_1163.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260000/A82164FA-F6EC-6845-B937-B590EAD0852A.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018C-UL2018_MiniAODv2-v1/out_489.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260000/FC2C4194-EFA3-F14E-9981-A957BCE15E2B.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018C-UL2018_MiniAODv2-v1/out_488.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260000/6E1092CB-9F5D-9540-8562-AB3FA064A559.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018C-UL2018_MiniAODv2-v1/out_487.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260000/65B9ADF2-6BC0-5145-8910-B0CF626E14AB.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018C-UL2018_MiniAODv2-v1/out_486.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/JetHT/MINIAOD/UL2018_MiniAODv2-v1/280000/F58AB66D-30AA-8C4F-83CC-478824FB1CC7.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018C-UL2018_MiniAODv2-v1/out_869.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260000/CC5A626D-99AD-1341-AB8C-DB2123EF59B4.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018C-UL2018_MiniAODv2-v1/out_484.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260000/81247B98-91D5-904D-BAB8-82604AE25214.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018C-UL2018_MiniAODv2-v1/out_483.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260000/7047C64B-B6D2-784B-93B7-8A32DCA9AA72.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018C-UL2018_MiniAODv2-v1/out_482.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260000/685D751D-0F12-7B4F-B4AC-556230D77D99.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018C-UL2018_MiniAODv2-v1/out_481.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260000/E6BCA71F-505A-3846-BBCE-EE0E30E795F6.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018C-UL2018_MiniAODv2-v1/out_480.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260000/92624754-30FF-E145-9C98-FC93F58D271F.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018C-UL2018_MiniAODv2-v1/out_509.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/JetHT/MINIAOD/UL2018_MiniAODv2-v1/40000/BF361C0D-DAF4-AF43-BEBF-E29D9694BF70.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018C-UL2018_MiniAODv2-v1/out_955.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260000/96C50D60-1F67-904E-83AA-95A1836288A2.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018C-UL2018_MiniAODv2-v1/out_472.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260000/2B860A57-B1BD-864B-9CE1-C0114EAD65FB.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018C-UL2018_MiniAODv2-v1/out_473.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260000/94D73564-88C3-F047-A65E-1E2C40459AFB.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018C-UL2018_MiniAODv2-v1/out_470.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260000/43959ADA-E8C9-6B4D-B38D-F7127899CB9E.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018C-UL2018_MiniAODv2-v1/out_471.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260000/37D8C6E6-55D9-D544-8660-7C864595829F.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018C-UL2018_MiniAODv2-v1/out_476.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260000/08DD77EF-B17A-1F4F-9DE2-3F0BF17AEB17.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018C-UL2018_MiniAODv2-v1/out_477.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260000/29622A87-3E14-1542-B28D-C15A69217578.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018C-UL2018_MiniAODv2-v1/out_474.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260000/0078F433-DAAE-7A40-A629-10354BBCFBB5.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018C-UL2018_MiniAODv2-v1/out_475.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/JetHT/MINIAOD/UL2018_MiniAODv2-v1/50000/11152EDA-3EAC-0945-9E32-AC1DF49DE6DD.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018C-UL2018_MiniAODv2-v1/out_46.root -a '-o ./ -d --year UL2018C'"
queue
arguments="-f /store/data/Run2018C/JetHT/MINIAOD/UL2018_MiniAODv2-v1/260000/FCF8FE90-ABE6-E940-B21C-1F993F82C599.root -o /eos/user/l/legao/NtupleStore/V0/2018/Data/JetHT_Run2018C-UL2018_MiniAODv2-v1/out_479.root -a '-o ./ -d --year UL2018C'"
queue
