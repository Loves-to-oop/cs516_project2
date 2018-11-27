echo "size, thrust, singlethread, multithread" > result.csv
echo "size, thrust" > thrust_result.csv
echo "size, singlethread" > singlethread_result.csv
echo "size, multithread" > multithread_result.csv

#i=0; while [ $i -le 5 ]; do echo $i; i=$((i+1)); done

#for ((i = 10; i <= 10000; i+=100))

i=10; while [ $i -le 10000 ]; 
do
	ts=$(date +%s%N) ; 

	./thrust $i 10 1
				
	tt=$((($(date +%s%N) - $ts)/1000000)) ;

	ts=$(date +%s%N) ; 

	./singlethread $i 10 1
				
	tt_bbp=$((($(date +%s%N) - $ts)/1000000)) ;

	ts=$(date +%s%N) ; 

	./multithread $i 10 1
				
	tt_qss=$((($(date +%s%N) - $ts)/1000000)) ;


	echo "Size: $i bbs: $tt ms, bbp: $tt_bbp ms, qss: $tt_qss ms"

	echo "$i, $tt, $tt_bbp, $tt_qss" >> result.csv

	echo "$i, $tt" >> thrust_result.csv
	echo "$i, $tt_bbp" >> singlethread_result.csv
	echo "$i, $tt_qss" >> multithread_result.csv
	i=$((i+100));
done
