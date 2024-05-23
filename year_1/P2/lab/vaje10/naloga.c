
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "naloga.h"

int poisciStudenta(Student **studentje, int stStudentov, int vpisna) {
  for (int i = 0; i < stStudentov; i++) {
    if (studentje[i]->vpisna == vpisna) {
      return i;
    }
  }

  return -1;
}

int poisciPO(Student *student, char *predmet) {
  for (int i = 0; i < student->stPO; i++) {
    if (strcmp((student->po)[i].predmet, predmet) == 0) {
      return i;
    }
  }

  return -1;
}

int dodaj(Student **studentje, int stStudentov, int vpisna, char *predmet,
          int ocena) {
  int index_studenta = poisciStudenta(studentje, stStudentov, vpisna);
  Student *student;

  if (index_studenta == -1) {
    student = studentje[stStudentov] = (Student *)malloc(sizeof(Student));
    student->vpisna = vpisna;
    student->stPO = 0;
    student->po = (PO *)malloc(10 * sizeof(PO));

    stStudentov++;
  } else {
    student = studentje[index_studenta];
  }

  int index_po = poisciPO(student, predmet);

  if (index_po == -1) {
    PO *po = student->po + student->stPO;
    strncpy(po->predmet, predmet, sizeof(po->predmet) - 1);
    (po->predmet)[sizeof(po->predmet) - 1] = '\0';
    po->ocena = ocena;

    student->stPO++;
  } else {
    (student->po + index_po)->ocena = ocena;
  }

  return stStudentov;
}

#ifndef test

int main() {
  // koda za ro"cno testiranje (po "zelji)

  return 0;
}

#endif
