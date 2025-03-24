# Makefile

CC = gcc
CFLAGS = -Wall -O2
TARGET = alexnet.exe

SRCDIR = 00_SW
OBJDIR = obj

SOURCES = $(wildcard $(SRCDIR)/*.c)
OBJECTS = $(patsubst $(SRCDIR)/%.c, $(OBJDIR)/%.o, $(SOURCES))

$(TARGET): $(OBJECTS)
	$(CC) $(CFLAGS) -o $@ $^

$(OBJDIR)/%.o: $(SRCDIR)/%.c
	@if not exist $(OBJDIR) mkdir $(OBJDIR)
	$(CC) $(CFLAGS) -c $< -o $@

clean:
	@if exist $(OBJDIR) rmdir /S /Q $(OBJDIR)
	@del /Q /F *.exe