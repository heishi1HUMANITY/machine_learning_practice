# COVID-19国内感染者数の回帰分析
COVID-19国内感染者数（2020/01/21 ~ 2020/08/07)をもとに行った。  
patient.csvは日付(date),  感染者数(total_confirmed_cases)が入っています。  
patient.pyは感染者数と予測関数がプロットされます。また、そこから単純に明日の感染者数を適当に出します。  
**このプログラムから出力されたデータを鵜呑みにしないでください。信憑性はありません**  
次数を増やして遊んでみてね！  
tmp.pyではscikit-learnを使って書いています、曜日も要素として追加してみました。余計だったかもしれません。  
[データ元はこちら](https://www.who.int/emergencies/diseases/novel-coronavirus-2019/situation-reports/)
