       IDENTIFICATION DIVISION.
       PROGRAM-ID. DAILYRPT.
      *> Nightly batch: reads customer master + account balances,
      *> writes a daily settlement report. Run by DAILYJOB.
       ENVIRONMENT DIVISION.
       INPUT-OUTPUT SECTION.
       FILE-CONTROL.
           SELECT CUSTOMER-FILE ASSIGN TO CUSTFILE
               ORGANIZATION IS INDEXED
               ACCESS MODE IS SEQUENTIAL
               RECORD KEY IS CUST-ID.
           SELECT REPORT-FILE ASSIGN TO RPTFILE
               ORGANIZATION IS SEQUENTIAL.
       DATA DIVISION.
       FILE SECTION.
       FD  CUSTOMER-FILE.
           COPY CUSTREC.
       FD  REPORT-FILE.
       01  REPORT-RECORD          PIC X(132).
       WORKING-STORAGE SECTION.
       01  WS-ACCT-BALANCE        PIC S9(11)V99 COMP-3.
       PROCEDURE DIVISION.
       MAIN-PARA.
           PERFORM UNTIL EOF
               READ CUSTOMER-FILE
                   AT END SET EOF TO TRUE
               END-READ
               EXEC SQL
                   SELECT ACCT_BALANCE INTO :WS-ACCT-BALANCE
                   FROM ACCOUNT
                   WHERE ACCT_CUST_ID = :CUST-ID
               END-EXEC
               WRITE REPORT-RECORD
           END-PERFORM.
           STOP RUN.
