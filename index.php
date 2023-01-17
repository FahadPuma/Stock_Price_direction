<!DOCTYPE html>
<!--
Click nbfs://nbhost/SystemFileSystem/Templates/Licenses/license-default.txt to change this license
Click nbfs://nbhost/SystemFileSystem/Templates/Scripting/EmptyPHPWebPage.php to edit this template
-->
<html>
    <head>
        <meta charset="UTF-8">
        <title>Stock Predictor</title>
        <h1 align="center">Stock Predictor</h1>
        <style>
table, th, td {
  border:1px solid black;
}
</style>
    </head>
    <body>
        <b> Enter the below details:</b>
        <br>
        <form action="" method="post">
            <br>
            <label for="sname">Stock Name:</label>
<select name="sname[]" id="stocks" multiple="multiple" required>
<option value="all">All</option>
<option value="ADANIPORTS.NS/Adani Ports">Adani Ports</option>
<option value="ASIANPAINT.NS/Asian Paints Limited">Asian Paints Limited</option>
<option value="AXISBANK.NS/Axis Bank Limited">Axis Bank Limited</option>
<option value="BAJAJ-AUTO.NS/Bajaj Auto Limited">Bajaj Auto Limited</option>
<option value="BAJAJFINSV.NS/Bajaj Finserv Limited">Bajaj Finserv Limited</option>
<option value="BAJFINANCE.NS/Bajaj Finance Limited">Bajaj Finance Limited</option>
<option value="BHARTIARTL.NS/Bharti Airtel Limited">Bharti Airtel Limited</option>
<option value="BPCL.NS/Bharat Petroleum Corporation Limited">Bharat Petroleum Corporation Limited</option>
<option value="BRITANNIA.NS/Britannia Industries Limited">Britannia Industries Limited</option>
<option value="CIPLA.NS/Cipla Limited">Cipla Limited</option>
<option value="COALINDIA.NS/Coal India Limited">Coal India Limited</option>
<option value="DIVISLAB.NS/Divis Laboratories Limited">Divis Laboratories Limited</option>
<option value="DRREDDY.NS/Dr. Reddys Laboratories Limited">Dr. Reddys Laboratories Limited</option>
<option value="EICHERMOT.NS/Eicher Motors Limited">Eicher Motors Limited</option>
<option value="GAIL.NS/GAIL (India) Limited">GAIL (India) Limited</option>
<option value="GRASIM.NS/Grasim Industries Limited">Grasim Industries Limited</option>
<option value="HCLTECH.NS/HCL Technologies Limited">HCL Technologies Limited</option>
<option value="HDFC.NS/Housing Development Finance Corporation Limited">Housing Development Finance Corporation Limited</option>
<option value="HDFCBANK.NS/HDFC Bank Limited">HDFC Bank Limited</option>
<option value="HDFCLIFE.NS/HDFC Life Insurance Company Limited">HDFC Life Insurance Company Limited</option>
<option value="HEROMOTOCO.NS/Hero MotoCorp Limited">Hero MotoCorp Limited</option>
<option value="HINDALCO.NS/Hindalco Industries Limited">Hindalco Industries Limited</option>
<option value="HINDUNILVR.NS/Hindustan Unilever Limited">Hindustan Unilever Limited</option>
<option value="ICICIBANK.NS/ICICI Bank Limited">ICICI Bank Limited</option>
<option value="INDUSINDBK.NS/IndusInd Bank Limited">IndusInd Bank Limited</option>
<option value="INFY.NS/Infosys Limited">Infosys Limited</option>
<option value="IOC.NS/Indian Oil Corporation Limited">Indian Oil Corporation Limited</option>
<option value="ITC.NS/ITC Limited">ITC Limited</option>
<option value="JSWSTEEL.NS/JSW Steel Limited">JSW Steel Limited</option>
<option value="KOTAKBANK.NS/Kotak Mahindra Bank Limited">Kotak Mahindra Bank Limited</option>
<option value="LT.NS/Larsen & Toubro Limited">Larsen & Toubro Limited</option>
<option value="M&M.NS/Mahindra & Mahindra Limited">Mahindra & Mahindra Limited</option>
<option value="MARUTI.NS/Maruti Suzuki India Limited">Maruti Suzuki India Limited</option>
<option value="NESTLEIND.NS/Nestle India Limited">Nestle India Limited</option>
<option value="NTPC.NS/NTPC Limited">NTPC Limited</option>
<option value="ONGC.NS/Oil & Natural Gas Corporation Limited">Oil & Natural Gas Corporation Limited</option>
<option value="POWERGRID.NS/Power Grid Corporation of India Limited">Power Grid Corporation of India Limited</option>
<option value="RELIANCE.NS/Reliance Industries Limited">Reliance Industries Limited</option>
<option value="SBILIFE.NS/SBI Life Insurance Company Limited">SBI Life Insurance Company Limited</option>
<option value="SBIN.NS/State Bank of India">State Bank of India</option>
<option value="SHREECEM.NS/Shree Cement Limited">Shree Cement Limited</option>
<option value="SUNPHARMA.NS/Sun Pharmaceutical Industries Limited">Sun Pharmaceutical Industries Limited</option>
<option value="TATAMOTORS.NS/Tata Motors Limited">Tata Motors Limited</option>
<option value="TATASTEEL.NS/Tata Steel Limited">Tata Steel Limited</option>
<option value="TCS.NS/Tata Consultancy Services Limited">Tata Consultancy Services Limited</option>
<option value="TECHM.NS/Tech Mahindra Limited">Tech Mahindra Limited</option>
<option value="TITAN.NS/Titan Company Limited">Titan Company Limited</option>
<option value="ULTRACEMCO.NS/UltraTech Cement Limited">UltraTech Cement Limited</option>
<option value="UPL.NS/UPL Limited">UPL Limited</option>
<option value="WIPRO.NS/Wipro Limited">Wipro Limited</option>
</select>
<br><br>
<label for="st_dt">Start Date:</label>
<input type="date" id="st_dt" name="st_dt" required><br><br>
<script type="text/javascript">
  document.getElementById('st_dt').value = "<?php echo $_POST['st_dt'];?>";
</script>
<label for="end_dt">End Date:</label>
<input type="date" id="end_dt" name="end_dt" required><br><br>
<script type="text/javascript">
  document.getElementById('end_dt').value = "<?php echo $_POST['end_dt'];?>";
</script>
<label for="intrvl">Interval:</label>
<select name="intrvl" id="intrvl" required>
<option value="1m">1 Minute</option>
<option value="2m">2 Minutes</option>
<option value="15m">15 Minutes</option>
<option value="30m">30 Minutes</option>
<option value="1h">1 Hour</option>
<option value="1d">1 Day</option>
<option value="1wk">1 Week</option>
<option value="1mo">1 Month</option>
</select>
<script type="text/javascript">
  document.getElementById('intrvl').value = "<?php echo $_POST['intrvl'];?>";
</script>
<br><br>
<input type="submit" name="Submit">
</form>
         
<?php
if(isset($_POST['Submit'])){ //check if form was submitted
  if(isset($_POST['sname'])){ 
    $stocks=$_POST['sname'];
    $len=count($stocks);
    echo       '<br><br>
<table style="width:100%">
  <tr>
    <th bgcolor="DeepSkyBlue">Stock Name</th>
    <th bgcolor="DeepSkyBlue">Close Price</th>
    <th bgcolor="DeepSkyBlue">Technical Analysis</th>
    <th bgcolor="DeepSkyBlue">Regression Analysis</th>
    <th bgcolor="DeepSkyBlue">Sentiment Analysis</th>
    <th bgcolor="DeepSkyBlue">Final Recommendation</th>
  </tr>';
if($stocks[0]=="all"){
  $stocks=array("ADANIPORTS.NS/Adani Ports",	"ASIANPAINT.NS/Asian Paints Limited",	"AXISBANK.NS/Axis Bank Limited",	"BAJAJ-AUTO.NS/Bajaj Auto Limited",	"BAJAJFINSV.NS/Bajaj Finserv Limited",	"BAJFINANCE.NS/Bajaj Finance Limited",	"BHARTIARTL.NS/Bharti Airtel Limited",	"BPCL.NS/Bharat Petroleum Corporation Limited",	"BRITANNIA.NS/Britannia Industries Limited",	"CIPLA.NS/Cipla Limited",	"COALINDIA.NS/Coal India Limited",	"DIVISLAB.NS/Divis Laboratories Limited",	"DRREDDY.NS/Dr. Reddys Laboratories Limited",	"EICHERMOT.NS/Eicher Motors Limited",	"GAIL.NS/GAIL (India) Limited",	"GRASIM.NS/Grasim Industries Limited",	"HCLTECH.NS/HCL Technologies Limited",	"HDFC.NS/Housing Development Finance Corporation Limited",	"HDFCBANK.NS/HDFC Bank Limited",	"HDFCLIFE.NS/HDFC Life Insurance Company Limited",	"HEROMOTOCO.NS/Hero MotoCorp Limited",	"HINDALCO.NS/Hindalco Industries Limited",	"HINDUNILVR.NS/Hindustan Unilever Limited",	"ICICIBANK.NS/ICICI Bank Limited",	"INDUSINDBK.NS/IndusInd Bank Limited",	"INFY.NS/Infosys Limited",	"IOC.NS/Indian Oil Corporation Limited",	"ITC.NS/ITC Limited",	"JSWSTEEL.NS/JSW Steel Limited",	"KOTAKBANK.NS/Kotak Mahindra Bank Limited",	"LT.NS/Larsen & Toubro Limited",	"M&M.NS/Mahindra & Mahindra Limited",	"MARUTI.NS/Maruti Suzuki India Limited",	"NESTLEIND.NS/Nestle India Limited",	"NTPC.NS/NTPC Limited",	"ONGC.NS/Oil & Natural Gas Corporation Limited",	"POWERGRID.NS/Power Grid Corporation of India Limited",	"RELIANCE.NS/Reliance Industries Limited",	"SBILIFE.NS/SBI Life Insurance Company Limited",	"SBIN.NS/State Bank of India",	"SHREECEM.NS/Shree Cement Limited",	"SUNPHARMA.NS/Sun Pharmaceutical Industries Limited",	"TATAMOTORS.NS/Tata Motors Limited",	"TATASTEEL.NS/Tata Steel Limited",	"TCS.NS/Tata Consultancy Services Limited",	"TECHM.NS/Tech Mahindra Limited",	"TITAN.NS/Titan Company Limited",	"ULTRACEMCO.NS/UltraTech Cement Limited",	"UPL.NS/UPL Limited",	"WIPRO.NS/Wipro Limited"); 
 for($i=0;$i<count($stocks);$i++){
$stock= $stocks[$i];
$pair = explode("/", $stock);
$st_dt=$_POST['st_dt'];
$end_dt=$_POST['end_dt'];
$intrvl=$_POST['intrvl'];
$cmd="python PredictionEngine.py ".$pair[0]." ".$st_dt." ".$end_dt." ".$intrvl ;
$command = escapeshellcmd($cmd);
           $output = shell_exec($command);
           $temp=explode(':',$output);
           $temp2=substr($temp[1],2,-3);
           $scores=explode(",",$temp2);
           $acc=(float)$scores[2]*100;
           $price=$scores[4];
           $final=0;
           if($scores[0]==$scores[1] && $scores[0]==$scores[3]){
               $final=$scores[0];
           }
           elseif($scores[0]==$scores[1] && $scores[0]!=$scores[3]){
               $final=$scores[0];
           }
           elseif($scores[0]!=$scores[1] && $scores[0]==$scores[3]){
               $final=$scores[0];
           }
           elseif($scores[0]!=$scores[1] && $scores[1]==$scores[3]){
               $final=$scores[1];
           }
           else{
               $final=2;
           }
           $ts=$rs=$ss=$fs="";
           if($scores[0]==0){$ts="SELL";}
           elseif($scores[0]==1){$ts="BUY";}
           else{$ts="HOLD";}
           if($scores[1]==0){$rs="SELL";}
           elseif($scores[1]==1){$rs="BUY";}
           else{$rs="HOLD";}
           if($scores[3]==0){$ss="Negative";}
           elseif($scores[3]==1){$ss="Positive";}
           else{$ss="Neutral";}
           if($final==0){$fs="SELL";}
           elseif($final==1){$fs="BUY";}
           else{$fs="HOLD";}
  $ct=$cr=$cs=$cf='';
    if($ts=="BUY"){$ct='green';}
    elseif($ts=="SELL"){$ct='red';}
    else{$ct='orange';}
    if($rs=="BUY"){$cr='green';}
    elseif($rs=="SELL"){$cr='red';}
    else{$cr='orange';}
    if($ss=="Positive"){$cs='green';}
    elseif($ss=="Negative"){$cs='red';}
    else{$cs='orange';}
    if($fs=="BUY"){$cf='green';}
    elseif($fs=="SELL"){$cf='red';}
    else{$cf='orange';}
    echo '<tr>
    <td style="text-align:center">'.$pair[1].'</td>
    <td style="text-align:center">'.$price.'</td>
    <td style="text-align:center;color:'.$ct.';">'.$ts.'</td> 
    <td style="text-align:center;color:'.$cr.';">'.$rs.'</td>
    <td style="text-align:center;color:'.$cs.';">'.$ss.'</td>
    <td style="text-align:center;color:'.$cf.';">'.$fs.'</td>
    </tr>';  
}
}
else{
for($i=0;$i<$len;$i++){
$stock= $stocks[$i];
$pair = explode("/", $stock);
$st_dt=$_POST['st_dt'];
$end_dt=$_POST['end_dt'];
$intrvl=$_POST['intrvl'];
$cmd="python PredictionEngine.py ".$pair[0]." ".$st_dt." ".$end_dt." ".$intrvl;
$command = escapeshellcmd($cmd);
           $output = shell_exec($command);
           $temp=explode(':',$output);
           $temp2=substr($temp[1],2,-3);
           $scores=explode(",",$temp2);
           $acc=(float)$scores[2]*100;
           $price=$scores[4];
           $final=0;
           if($scores[0]==$scores[1] && $scores[0]==$scores[3]){
               $final=$scores[0];
           }
           elseif($scores[0]==$scores[1] && $scores[0]!=$scores[3]){
               $final=$scores[0];
           }
           elseif($scores[0]!=$scores[1] && $scores[0]==$scores[3]){
               $final=$scores[0];
           }
           elseif($scores[0]!=$scores[1] && $scores[1]==$scores[3]){
               $final=$scores[1];
           }
           else{
               $final=2;
           }
           $ts=$rs=$ss=$fs="";
           if($scores[0]==0){$ts="SELL";}
           elseif($scores[0]==1){$ts="BUY";}
           else{$ts="HOLD";}
           if($scores[1]==0){$rs="SELL";}
           elseif($scores[1]==1){$rs="BUY";}
           else{$rs="HOLD";}
           if($scores[3]==0){$ss="Negative";}
           elseif($scores[3]==1){$ss="Positive";}
           else{$ss="Neutral";}
           if($final==0){$fs="SELL";}
           elseif($final==1){$fs="BUY";}
           else{$fs="HOLD";}
    $ct=$cr=$cs=$cf='';
    if($ts=="BUY"){$ct='green';}
    elseif($ts=="SELL"){$ct='red';}
    else{$ct='orange';}
    if($rs=="BUY"){$cr='green';}
    elseif($rs=="SELL"){$cr='red';}
    else{$cr='orange';}
    if($ss=="Positive"){$cs='green';}
    elseif($ss=="Negative"){$cs='red';}
    else{$cs='orange';}
    if($fs=="BUY"){$cf='green';}
    elseif($fs=="SELL"){$cf='red';}
    else{$cf='orange';}
    echo '<tr>
    <td style="text-align:center">'.$pair[1].'</td>
    <td style="text-align:center">'.$price.'</td>
    <td style="text-align:center;color:'.$ct.';">'.$ts.'</td> 
    <td style="text-align:center;color:'.$cr.';">'.$rs.'</td>
    <td style="text-align:center;color:'.$cs.';">'.$ss.'</td>
    <td style="text-align:center;color:'.$cf.';">'.$fs.'</td>
    </tr>';
}
}
echo '</table>';
}
}
?>
</body>
</html>
