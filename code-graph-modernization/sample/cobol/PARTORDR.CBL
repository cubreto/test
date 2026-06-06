       IDENTIFICATION DIVISION.
       PROGRAM-ID. PARTORDR.
      *> Online parts ordering transaction (screen-driven, DB2).
      *> Independent functional area from customer/account management.
       ENVIRONMENT DIVISION.
       DATA DIVISION.
       WORKING-STORAGE SECTION.
           COPY PARTREC.
       01  WS-PART-ID         PIC 9(6).
       01  WS-ORDER-QTY       PIC 9(5).
       PROCEDURE DIVISION.
       MAIN-PARA.
           DISPLAY "PARTS ORDER ENTRY".
           DISPLAY "ENTER PART ID:".
           ACCEPT WS-PART-ID.
           ACCEPT WS-ORDER-QTY.
           EXEC SQL
               SELECT PART_QTY_ON_HAND INTO :PART-QTY-ON-HAND
               FROM PARTS
               WHERE PART_ID = :WS-PART-ID
           END-EXEC.
           EXEC SQL
               UPDATE PARTS
               SET PART_QTY_ON_HAND = :PART-QTY-ON-HAND
               WHERE PART_ID = :WS-PART-ID
           END-EXEC.
           EXEC SQL
               INSERT INTO ORDERS
                   (ORD_PART_ID, ORD_QTY)
               VALUES (:WS-PART-ID, :WS-ORDER-QTY)
           END-EXEC.
           STOP RUN.
