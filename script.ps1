#---might be neccesary to run this PS script---
#Set-ExecutionPolicy -ExecutionPolicy Bypass -Scope CurrentUser



#---organization of the data---
<#
New-Item -ItemType Directory -Force -Path C:\images

$filesperfolder = 1
$sourcePath = "C:\Users\ja\fiftyone\open-images-v6\validation\data"
$destPath = "C:\images"
$i = 0
$folderNum = 0
Get-ChildItem "$sourcePath\*.jpg" | % {
    New-Item -Path ($destPath + "\" + $folderNum) -Type Directory -Force
    Copy-Item $_ ($destPath + "\" + $folderNum)
    $i++
    if ($i -eq $filesperfolder){
        $folderNum++
        $i = 0 
    }
}
Get-ChildItem -Path "C:\images\*.jpg" -Recurse | Move-Item -Destination "C:\Users\ja\fiftyone\open-images-v6\validation\data"

$sourcePath = "C:\Users\ja\fiftyone\open-images-v6\validation\data"
$destPath = "C:\images"
$i = 0
Get-ChildItem "$sourcePath\*.jpg" | % {
    New-Item -Path ($destPath + "\" + $i) -Type Directory -Force
    Copy-Item $_ ($destPath + "\" + $i)
    if ($i -eq 999){
        break
    }
    #Write-Host $i
    $i++
}

$sourcePath = "C:\images"
$i = 0
Get-ChildItem "C:\images" | ForEach-Object{
    New-Item -Path ($sourcePath + "\" + $i + "\" + "data") -Type Directory -Force
    Write-Host $i
    $i++
}
#>



#---runs modified YOLO (crashproofed)---

$sourcePath = "C:\images"
clear
cd C:\Yolo2017\darknet-master\build\darknet\x64
for($i = 0; $i -lt 1000; $i++){
    $picture = Get-ChildItem -Path "$sourcePath\${i}\*.jpg"
    Write-Host $picture
    while($true){
        try{
            .\darknet.exe detect cfg\yolov3.cfg yolov3.weights ${picture}
            $x = $LastExitCode
        }
        catch{
            $x = $?
        }
        Write-Host ("EXIT CODE: " + $x)
        if ($x -eq 0){
            break
        }
    }
    Move-Item -Path "C:\Yolo2017\darknet-master\build\darknet\x64\original.csv" -Destination "$sourcePath\$i\data"
    Move-Item -Path "C:\Yolo2017\darknet-master\build\darknet\x64\vanillaYOLOresults.csv" -Destination "$sourcePath\$i\data"
    Move-Item -Path "C:\Yolo2017\darknet-master\build\darknet\x64\predictionsO.jpg" -Destination "$sourcePath\$i\data"
    #---necessary when YOLO runs only for one type of picture---
    <#
    while($true){
        try{
            .\darknet.exe detect cfg\yolov3.cfg yolov3.weights ${picture}
            $x = $LastExitCode
        }
        catch{
            $x = $?
        }
        Write-Host ("EXIT CODE: " + $x)
        if ($x -eq 0){
            break
        }
        #break
    }
    #>
    Move-Item -Path "C:\Yolo2017\darknet-master\build\darknet\x64\mirror.csv" -Destination "$sourcePath\$i\data"
    Move-Item -Path "C:\Yolo2017\darknet-master\build\darknet\x64\predictionsM.jpg" -Destination "$sourcePath\$i\data"
    #---necessary when YOLO runs only for one type of picture---
    <#
    while($true){
        try{
            .\darknet.exe detect cfg\yolov3.cfg yolov3.weights ${picture}
            $x = $LastExitCode
        }
        catch{
            $x = $?
        }
        Write-Host ("EXIT CODE: " + $x)
        if ($x -eq 0){
            break
        }
    }
    #>
    Move-Item -Path "C:\Yolo2017\darknet-master\build\darknet\x64\rotated.csv" -Destination "$sourcePath\$i\data"
    Move-Item -Path "C:\Yolo2017\darknet-master\build\darknet\x64\rotated.jpg" -Destination "$sourcePath\$i\data"
    Move-Item -Path "C:\Yolo2017\darknet-master\build\darknet\x64\predictionsR.jpg" -Destination "$sourcePath\$i\data"

    clear
}


#---runs enhancing program---

$sourcePath = "C:\images"
cd C:\Users\ja\PycharmProjects\YOLOaddition
for($i = 0; $i -lt 1000; $i++){
    #clear
    $picture = Get-ChildItem -Name "$sourcePath\${i}\*.jpg"
    Write-Host $i
    python main.py ${sourcePath} ${picture} ${i}
    Move-Item -Path "C:\Users\ja\PycharmProjects\YOLOaddition\final.jpg" -Destination "$sourcePath\$i\data"
    Move-Item -Path "C:\Users\ja\PycharmProjects\YOLOaddition\final_resultsV1.csv" -Destination "$sourcePath\$i\data"
    Move-Item -Path "C:\Users\ja\PycharmProjects\YOLOaddition\ground_truths.csv" -Destination "$sourcePath\$i\data"
    Move-Item -Path "C:\Users\ja\PycharmProjects\YOLOaddition\YOLO.csv" -Destination "$sourcePath\$i\data"
    Move-Item -Path "C:\Users\ja\PycharmProjects\YOLOaddition\final_resultsV2.csv" -Destination "$sourcePath\$i\data"
    Move-Item -Path "C:\Users\ja\PycharmProjects\YOLOaddition\final_resultsV3.csv" -Destination "$sourcePath\$i\data"
}


#---runs program that calculates statistics---

$sourcePath = "C:\images"
clear
cd C:\Users\ja\PycharmProjects\resultsAggregation
python main.py ${sourcePath} "final_resultsV1.csv" "1"
python main.py ${sourcePath} "final_resultsV2.csv" "2"
python main.py ${sourcePath} "final_resultsV3.csv" "3"


#---removes all recognition data---
<#
$sourcePath = "C:\images"
for($i = 0; $i -lt 1000; $i++){
    Write-Host $i
    Remove-Item "$sourcePath\$i\data\final.jpg"
    Remove-Item "$sourcePath\$i\data\ground_truths.csv"
    Remove-Item "$sourcePath\$i\data\YOLO.csv"
    Remove-Item "$sourcePath\$i\data\final_resultsV1.csv"
    Remove-Item "$sourcePath\$i\data\final_resultsV2.csv"
    Remove-Item "$sourcePath\$i\data\final_resultsV3.csv"
    Remove-Item "$sourcePath\$i\data\summary1.csv"
    Remove-Item "$sourcePath\$i\data\summary2.csv"
    Remove-Item "$sourcePath\$i\data\summary3.csv"
    Remove-Item "$sourcePath\totalSummary1.csv"
    Remove-Item "$sourcePath\totalSummary2.csv"
    Remove-Item "$sourcePath\totalSummary3.csv"
    Remove-Item "$sourcePath\statistics1.txt"
    Remove-Item "$sourcePath\statistics2.txt"
    Remove-Item "$sourcePath\statistics3.txt"
}
#>

#Start-Sleep -s 3
#Read-Host -Prompt "press anything to finish"