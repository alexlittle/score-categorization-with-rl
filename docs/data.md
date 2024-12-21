# Data Descriptions


## data/course_bbb_2013b_train.csv

Filtered version of OULAD studentassessment to only the 2013b presentation of the course.

* Num of students: 1369
* Total num rows: 11256

Include the total number of distinct activities from the studentvle table that the student prior to the date of the
assessment. 

SQL query used to generate the dataset:

SELECT sa.*, COUNT(DISTINCT sv.id_site) AS total_vle_before_assessment 
FROM studentassessment sa LEFT JOIN studentvle sv ON sv.id_student = sa.id_student AND sv.date < sa.date_submitted 
WHERE sa.id_assessment in (select id_assessment from assessments where code_module="BBB" and code_presentation="2013B") 
GROUP BY sa.id;



## data/course_bbb_2013j_test1.csv

As above, only filtered for the 2013J presentation of the course

* Num of students:
* Total num rows:



## data/course_bbb_2014b_test2.csv

As above, only filtered for the 2014B presentation of the course

* Num of students:
* Total num rows:


## data/demographics.csv




## data/demographics_encoded.csv