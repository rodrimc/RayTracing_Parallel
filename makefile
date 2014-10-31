app = RayTracing

utillib = libutil.a
utilc = util.cpp
utilo = util.o
srcExt = cpp
srcDir = .
objDir = build
binDir = build
inc = inc
output = out.ppm
main = main.cpp
main_openmp = main_openmp.cpp
main_cuda = main_cuda.cpp

debug = 0

CFlags_common = -Wall -std=c++0x
CFlags_openmp = -fopenmp
CFlags_cuda =
LDFlags = -fopenmp
libs = 


#************************ DO NOT EDIT BELOW THIS LINE! ************************
ifeq ($(debug),1)
	debug=-g
else
	debug=
endif
inc := $(addprefix -I,$(inc))
libs := $(addprefix -l,$(libs))
libDir := $(addprefix -L,$(libDir))
CFlags = $(CFlags_common) $(debug) $(inc) $(libDir) $(libs) 
srcDirs := $(shell find . -name '*.$(srcExt)' -exec dirname {} \; | uniq)
objects := $(patsubst %.$(srcExt),$(objDir)/%.o,$(sources))

ifeq ($(srcExt),cpp)
	CC = $(CXX)
else
	CFlags += -std=gnu99
endif

.phony: all clean distclean

all: $(binDir)/$(utillib) $(binDir)/$(app) $(binDir)/$(app)_openmp 
	
#$(binDir)/$(app)_cuda

$(binDir)/$(utillib):
	@mkdir -p `dirname $@`
	@echo "Compiling $(utillib)"
	@$(CC) -c $(utilc) $(CFlags)
	@ar rcs $(binDir)/$(utillib) $(utilo)

$(binDir)/$(app): $(main)
	@echo "Compiling $@..."
	@$(CC) $(main) $(CFlags) $(LDFlags) -o $@ -static $(binDir)/$(utillib)

$(binDir)/$(app)_openmp: $(main_openmp)
	@echo "Compiling $@..."
	@$(CC) $(main_openmp) $(CFlags) $(CFlags_openmp) $(LDFlags) -o $@ -static $(binDir)/$(utillib)

$(binDir)/$(app)_cuda: $(main_cuda)
	@echo "Compiling $@..."
	@$(CC) $(main_cuda) $(CFlags) $(CFlags_cuda)$(LDFlags) -o $@ -static $(binDir)/$(utillib)

$(objDir)/%.o: %.$(srcExt)
	@echo "Generating dependencies for $<..."
	@$(call make-depend,$<,$@,$(subst .o,.d,$@))
	@echo "Compiling $<..."
	@$(CC) $(CFlags) $< -o $@

clean:
	$(RM) $(output)
	$(RM) -r $(objDir)

distclean: clean
	$(RM) $(output)
	$(RM) -r $(binDir)

buildrepo:
	@$(call make-repo)

define make-repo
   for dir in $(srcDirs); \
   do \
	mkdir -p $(objDir)/$$dir; \
   done
endef

# usage: $(call make-depend,source-file,object-file,depend-file)
define make-depend
  $(CC) -MM       \
        -MF $3    \
        -MP       \
        -MT $2    \
        $(CFlags) \
        $1
endef
