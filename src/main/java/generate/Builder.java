/*
 * Copyright (C) 2011-2018 Samuel Audet
 *
 * Licensed either under the Apache License, Version 2.0, or (at your option)
 * under the terms of the GNU General Public License as published by
 * the Free Software Foundation (subject to the "Classpath" exception),
 * either version 2, or any later version (collectively, the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *     http://www.gnu.org/licenses/
 *     http://www.gnu.org/software/classpath/license.html
 *
 * or as provided in the LICENSE.txt file that accompanied this code.
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package generate;

import org.bytedeco.javacpp.ClassProperties;
import org.bytedeco.javacpp.Loader;
import org.bytedeco.javacpp.tools.*;

import java.io.*;
import java.net.URI;
import java.net.URISyntaxException;
import java.net.URL;
import java.nio.file.*;
import java.nio.file.attribute.BasicFileAttributes;
import java.util.*;
import java.util.jar.JarOutputStream;
import java.util.zip.ZipEntry;

/**
 * The Builder is responsible for coordinating efforts between the Parser, the
 * Generator, and the native compiler. It contains the main() method, and basically
 * takes care of the tasks one would expect from a command line build tool, but
 * can also be used programmatically by setting its properties and calling build().
 *
 * @author Samuel Audet
 */
public class Builder {

    /**
     * Calls {@link Parser#parse(File, String[], Class)} after creating an instance of the Class.
     *
     * @param classPath an array of paths to try to load header files from
     * @param cls The class annotated with {@link org.bytedeco.javacpp.annotation.Properties}
     *            and implementing {@link InfoMapper}
     * @return the target File produced
     * @throws IOException on Java target file writing error
     * @throws ParserException on C/C++ header file parsing error
     */
    File parse(String[] classPath, Class cls) throws IOException, ParserException {
        return new Parser(logger, properties, encoding, null).parse(outputDirectory, classPath, cls);
    }

    /**
     * Tries to find automatically include paths for {@code jni.h} and {@code jni_md.h},
     * as well as the link and library paths for the {@code jvm} library.
     *
     * @param properties the Properties containing the paths to update
     * @param header to request support for exporting callbacks via generated header file
     */
    void includeJavaPaths(ClassProperties properties, boolean header) {
        if (properties.getProperty("platform", "").startsWith("android")) {
            // Android includes its own jni.h file and doesn't have a jvm library
            return;
        }
        String platform = Loader.getPlatform();
        final String jvmlink = properties.getProperty("platform.link.prefix", "") +
                       "jvm" + properties.getProperty("platform.link.suffix", "");
        final String jvmlib  = properties.getProperty("platform.library.prefix", "") +
                       "jvm" + properties.getProperty("platform.library.suffix", "");
        final String[] jnipath = new String[2];
        final String[] jvmpath = new String[2];
        FilenameFilter filter = new FilenameFilter() {
            @Override public boolean accept(File dir, String name) {
                if (new File(dir, "jni.h").exists()) {
                    jnipath[0] = dir.getAbsolutePath();
                }
                if (new File(dir, "jni_md.h").exists()) {
                    jnipath[1] = dir.getAbsolutePath();
                }
                if (new File(dir, jvmlink).exists()) {
                    jvmpath[0] = dir.getAbsolutePath();
                }
                if (new File(dir, jvmlib).exists()) {
                    jvmpath[1] = dir.getAbsolutePath();
                }
                return new File(dir, name).isDirectory();
            }
        };
        File javaHome;
        try {
            javaHome = new File(System.getProperty("java.home")).getParentFile().getCanonicalFile();
        } catch (IOException | NullPointerException e) {
            logger.warn("Could not include header files from java.home:" + e);
            return;
        }
        ArrayList<File> dirs = new ArrayList<File>(Arrays.asList(javaHome.listFiles(filter)));
        while (!dirs.isEmpty()) {
            File d = dirs.remove(dirs.size() - 1);
            String dpath = d.getPath();
            File[] files = d.listFiles(filter);
            if (dpath == null || files == null) {
                continue;
            }
            for (File f : files) {
                try {
                    f = f.getCanonicalFile();
                } catch (IOException e) { }
                if (!dpath.startsWith(f.getPath())) {
                    dirs.add(f);
                }
            }
        }
        if (jnipath[0] != null && jnipath[0].equals(jnipath[1])) {
            jnipath[1] = null;
        } else if (jnipath[0] == null) {
            String macpath = "/System/Library/Frameworks/JavaVM.framework/Headers/";
            if (new File(macpath).isDirectory()) {
                jnipath[0] = macpath;
            }
        }
        if (jvmpath[0] != null && jvmpath[0].equals(jvmpath[1])) {
            jvmpath[1] = null;
        }
        properties.addAll("platform.includepath", jnipath);
        if (platform.equals(properties.getProperty("platform", platform))) {
            if (header) {
                // We only need libjvm for callbacks exported with the header file
                properties.get("platform.link").add(0, "jvm");
                properties.addAll("platform.linkpath", jvmpath);
            }
            if (platform.startsWith("macosx")) {
                properties.addAll("platform.framework", "JavaVM");
            }
        }
    }



    /**
     * Generates a C++ source file for classes, and compiles everything in
     * one shared library when {@code compile == true}.
     *
     * @param classes the Class objects as input to Generator
     * @param outputName the output name of the shared library
     * @return the actual File generated, either the compiled library or its source
     * @throws IOException
     * @throws InterruptedException
     */
    boolean generate(Class[] classes, String outputName, boolean first, boolean last) throws IOException, InterruptedException {
        File outputPath = outputDirectory != null ? outputDirectory.getCanonicalFile() : null;
        ClassProperties p = Loader.loadProperties(classes, properties, true);
        String platform     = properties.getProperty("platform");
        String extension    = properties.getProperty("platform.extension");
        String sourcePrefix = outputPath != null ? outputPath.getPath() + File.separator : "";
        String sourceSuffix = p.getProperty("platform.source.suffix", ".cpp");
        String libraryPath  = p.getProperty("platform.library.path", "");
        String libraryPrefix  = p.getProperty("platform.library.prefix", "") ;
        String librarySuffix  = p.getProperty("platform.library.suffix", "");
        String[] sourcePrefixes = {sourcePrefix, sourcePrefix};
        if (outputPath == null) {
            URI uri = null;
            try {
                String resourceName = '/' + classes[classes.length - 1].getName().replace('.', '/')  + ".class";
                String resourceURL = classes[classes.length - 1].getResource(resourceName).toString();
                uri = new URI(resourceURL.substring(0, resourceURL.lastIndexOf('/') + 1));
                boolean isFile = "file".equals(uri.getScheme());
                File classPath = new File(classScanner.getClassLoader().getPaths()[0]).getCanonicalFile();
                // If our class is not a file, use first path of the user class loader as base for our output path
                File packageDir = isFile ? new File(uri)
                                         : new File(classPath, resourceName.substring(0, resourceName.lastIndexOf('/') + 1));
                // Output to the library path inside of the class path, if provided by the user
                uri = new URI(resourceURL.substring(0, resourceURL.length() - resourceName.length() + 1));
                File targetDir = libraryPath.length() > 0
                        ? (isFile ? new File(uri) : classPath)
                        : new File(packageDir, platform + (extension != null ? extension : ""));
                outputPath = new File(targetDir, libraryPath);
                sourcePrefix = packageDir.getPath() + File.separator;
                // make sure jnijavacpp.cpp ends up in the same directory for all classes in different packages
                sourcePrefixes = new String[] {classPath.getPath() + File.separator, sourcePrefix};
            } catch (URISyntaxException e) {
                throw new RuntimeException(e);
            } catch (IllegalArgumentException e) {
                throw new RuntimeException("URI: " + uri, e);
            }
        }
        if (!outputPath.exists()) {
            outputPath.mkdirs();
        }
        Generator generator = new Generator(logger, properties, encoding);
        String[] sourceFilenames = {sourcePrefixes[0] + "jnijavacpp" + sourceSuffix,
                                    sourcePrefixes[1] + outputName + sourceSuffix};
        String[] headerFilenames = {null, header ? sourcePrefixes[1] + outputName +  ".h" : null};
        String[] loadSuffixes = {"_jnijavacpp", null};
        String[] baseLoadSuffixes = {null, "_jnijavacpp"};
        String classPath = System.getProperty("java.class.path");
        for (String s : classScanner.getClassLoader().getPaths()) {
            classPath += File.pathSeparator + s;
        }
        String[] classPaths = {null, classPath};
        Class[][] classesArray = {null, classes};

        boolean generated = true;
        for (int i = 0; i < sourceFilenames.length; i++) {
            if (i == 0 && !first) {
                continue;
            }
            logger.info("Generating " + sourceFilenames[i]);
            if (!generator.generate(sourceFilenames[i], headerFilenames[i],
                    loadSuffixes[i], baseLoadSuffixes[i], classPaths[i], classesArray[i])) {
                logger.info("Nothing generated for " + sourceFilenames[i]);
                generated = false;
                break;
            }
        }

        return generated;

    }



    /**
     * Default constructor that simply initializes everything.
     */
    public Builder() {
        this(Logger.create(Builder.class));
    }
    /**
     * Constructor that simply initializes everything.
     * @param logger where to send messages
     */
    public Builder(Logger logger) {
        this.logger = logger;
        System.setProperty("org.bytedeco.javacpp.loadlibraries", "false");
        properties = Loader.loadProperties();
        classScanner = new ClassScanner(logger, new ArrayList<Class>(),
                new UserClassLoader(Thread.currentThread().getContextClassLoader()));
        compilerOptions = new ArrayList<String>();
    }

    /** Logger where to send debug, info, warning, and error messages. */
    final Logger logger;
    /** The name of the character encoding used for input files as well as output files. */
    String encoding = null;
    /** The directory where the generated files and compiled shared libraries get written to.
     *  By default they are placed in the same directory as the {@code .class} file. */
    File outputDirectory = null;
    /** The name of the output generated source file or shared library. This enables single-
     *  file output mode. By default, the top-level enclosing classes get one file each. */
    String outputName = null;
    /** The name of the JAR file to create, if not {@code null}. */
    String jarPrefix = null;
    /** If true, compiles the generated source file to a shared library and deletes source. */
    boolean compile = true;
    /** If true, preserves the generated C++ JNI files after compilation */
    boolean deleteJniFiles = true;
    /** If true, also generates C++ header files containing declarations of callback functions. */
    boolean header = false;
    /** If true, also copies to the output directory dependent shared libraries (link and preload). */
    boolean copyLibs = false;
    /** If true, also copies to the output directory resources listed in properties. */
    boolean copyResources = false;
    /** Accumulates the various properties loaded from resources, files, command line options, etc. */
    Properties properties = null;
    /** The instance of the {@link ClassScanner} that fills up a {@link Collection} of {@link Class} objects to process. */
    ClassScanner classScanner = null;
    /** A system command for {@link ProcessBuilder} to execute for the build, instead of JavaCPP itself. */
    String[] buildCommand = null;
    /** User specified working directory to execute build subprocesses under. */
    File workingDirectory = null;
    /** User specified environment variables to pass to the native compiler. */
    Map<String,String> environmentVariables = null;
    /** Contains additional command line options from the user for the native compiler. */
    Collection<String> compilerOptions = null;

    /** Splits argument with {@link File#pathSeparator} and appends result to paths of the {@link #classScanner}. */
    public Builder classPaths(String classPaths) {
        classPaths(classPaths == null ? null : classPaths.split(File.pathSeparator));
        return this;
    }
    /** Appends argument to the paths of the {@link #classScanner}. */
    public Builder classPaths(String ... classPaths) {
        classScanner.getClassLoader().addPaths(classPaths);
        return this;
    }
    /** Sets the {@link #encoding} field to the argument. */
    public Builder encoding(String encoding) {
        this.encoding = encoding;
        return this;
    }
    /** Sets the {@link #outputDirectory} field to the argument. */
    public Builder outputDirectory(String outputDirectory) {
        outputDirectory(outputDirectory == null ? null : new File(outputDirectory));
        return this;
    }
    /** Sets the {@link #outputDirectory} field to the argument. */
    public Builder outputDirectory(File outputDirectory) {
        this.outputDirectory = outputDirectory;
        return this;
    }
    /** Sets the {@link #compile} field to the argument. */
    public Builder compile(boolean compile) {
        this.compile = compile;
        return this;
    }
    /** Sets the {@link #deleteJniFiles} field to the argument. */
    public Builder deleteJniFiles(boolean deleteJniFiles) {
        this.deleteJniFiles = deleteJniFiles;
        return this;
    }
    /** Sets the {@link #header} field to the argument. */
    public Builder header(boolean header) {
        this.header = header;
        return this;
    }
    /** Sets the {@link #copyLibs} field to the argument. */
    public Builder copyLibs(boolean copyLibs) {
        this.copyLibs = copyLibs;
        return this;
    }
    /** Sets the {@link #copyResources} field to the argument. */
    public Builder copyResources(boolean copyResources) {
        this.copyResources = copyResources;
        return this;
    }
    /** Sets the {@link #outputName} field to the argument. */
    public Builder outputName(String outputName) {
        this.outputName = outputName;
        return this;
    }
    /** Sets the {@link #jarPrefix} field to the argument. */
    public Builder jarPrefix(String jarPrefix) {
        this.jarPrefix = jarPrefix;
        return this;
    }
    /** Sets the {@link #properties} field to the ones loaded from resources for the specified platform. */
    public Builder properties(String platform) {
        if (platform != null) {
            properties = Loader.loadProperties(platform, null);
        }
        return this;
    }
    /** Adds all the properties of the argument to the {@link #properties} field. */
    public Builder properties(Properties properties) {
        if (properties != null) {
            for (Map.Entry e : properties.entrySet()) {
                property((String)e.getKey(), (String)e.getValue());
            }
        }
        return this;
    }
    /** Sets the {@link #properties} field to the ones loaded from the specified file. */
    public Builder propertyFile(String filename) throws IOException {
        propertyFile(filename == null ? null : new File(filename));
        return this;
    }
    /** Sets the {@link #properties} field to the ones loaded from the specified file. */
    public Builder propertyFile(File propertyFile) throws IOException {
        if (propertyFile == null) {
            return this;
        }
        FileInputStream fis = new FileInputStream(propertyFile);
        properties = new Properties();
        try {
            properties.load(new InputStreamReader(fis));
        } catch (NoSuchMethodError e) {
            properties.load(fis);
        }
        fis.close();
        return this;
    }
    /** Sets a property of the {@link #properties} field, in either "key=value" or "key:value" format. */
    public Builder property(String keyValue) {
        int equalIndex = keyValue.indexOf('=');
        if (equalIndex < 0) {
            equalIndex = keyValue.indexOf(':');
        }
        property(keyValue.substring(2, equalIndex),
                 keyValue.substring(equalIndex+1));
        return this;
    }
    /** Sets a key/value pair property of the {@link #properties} field. */
    public Builder property(String key, String value) {
        if (key.length() > 0 && value.length() > 0) {
            properties.put(key, value);
        }
        return this;
    }
    /** Requests the {@link #classScanner} to add a class or all classes from a package.
     *  A {@code null} argument indicates the unnamed package. */
    public Builder classesOrPackages(String ... classesOrPackages) throws IOException, ClassNotFoundException, NoClassDefFoundError {
        if (classesOrPackages == null) {
            classScanner.addPackage(null, true);
        } else for (String s : classesOrPackages) {
            classScanner.addClassOrPackage(s);
        }
        return this;
    }
    /** Sets the {@link #buildCommand} field to the argument. */
    public Builder buildCommand(String[] buildCommand) {
        this.buildCommand = buildCommand;
        return this;
    }
    /** Sets the {@link #workingDirectory} field to the argument. */
    public Builder workingDirectory(String workingDirectory) {
        workingDirectory(workingDirectory == null ? null : new File(workingDirectory));
        return this;
    }
    /** Sets the {@link #workingDirectory} field to the argument. */
    public Builder workingDirectory(File workingDirectory) {
        this.workingDirectory = workingDirectory;
        return this;
    }
    /** Sets the {@link #environmentVariables} field to the argument. */
    public Builder environmentVariables(Map<String,String> environmentVariables) {
        this.environmentVariables = environmentVariables;
        return this;
    }
    /** Appends arguments to the {@link #compilerOptions} field. */
    public Builder compilerOptions(String ... options) {
        if (options != null) {
            compilerOptions.addAll(Arrays.asList(options));
        }
        return this;
    }

    /**
     * Starts the build process and returns an array of {@link File} produced.
     *
     * @return the array of File produced
     * @throws IOException
     * @throws InterruptedException
     * @throws ParserException
     */
    public boolean build() throws IOException, InterruptedException, ParserException {
        if (buildCommand != null && buildCommand.length > 0) {
            List<String> command = Arrays.asList(buildCommand);
            String platform  = Loader.getPlatform();
            boolean windows = platform.startsWith("windows");
            for (int i = 0; i < command.size(); i++) {
                String arg = command.get(i);
                if (arg == null) {
                    arg = "";
                }
                if (arg.trim().isEmpty() && windows) {
                    // seems to be the only way to pass empty arguments on Windows?
                    arg = "\"\"";
                }
                command.set(i, arg);
            }

            String text = "";
            for (String s : command) {
                boolean hasSpaces = s.indexOf(" ") > 0 || s.isEmpty();
                if (hasSpaces) {
                    text += windows ? "\"" : "'";
                }
                text += s;
                if (hasSpaces) {
                    text += windows ? "\"" : "'";
                }
                text += " ";
            }
            logger.info(text);

            ProcessBuilder pb = new ProcessBuilder(command);
            if (workingDirectory != null) {
                pb.directory(workingDirectory);
            }
            if (environmentVariables != null) {
                pb.environment().putAll(environmentVariables);
            }
            String paths = properties.getProperty("platform.buildpath", "");
            String links = properties.getProperty("platform.linkresource", "");
            String resources = properties.getProperty("platform.buildresource", "");
            String separator = properties.getProperty("platform.path.separator");
            if (paths.length() > 0 || resources.length() > 0) {

                // Get all native libraries for classes on the class path.
                List<String> libs = new ArrayList<String>();
                ClassProperties libProperties = null;
                for (Class c : classScanner.getClasses()) {
                    if (Loader.getEnclosingClass(c) != c) {
                        continue;
                    }
                    libProperties = Loader.loadProperties(c, properties, true);
                    if (!libProperties.isLoaded()) {
                        logger.warn("Could not load platform properties for " + c);
                        continue;
                    }
                    libs.addAll(libProperties.get("platform.preload"));
                    libs.addAll(libProperties.get("platform.link"));
                }
                if (libProperties == null) {
                    libProperties = new ClassProperties(properties);
                }

                // Extract the required resources.
                for (String s : resources.split(separator)) {
                    for (File f : Loader.cacheResources(s)) {
                        String path = f.getCanonicalPath();
                        if (paths.length() > 0 && !paths.endsWith(separator)) {
                            paths += separator;
                        }
                        paths += path;

                        // Also create symbolic links for native libraries found there.
                        List<String> linkPaths = new ArrayList<String>();
                        for (String s2 : links.split(separator)) {
                            for (File f2 : Loader.cacheResources(s2)) {
                                String path2 = f2.getCanonicalPath();
                                if (path2.startsWith(path) && !path2.equals(path)) {
                                    linkPaths.add(path2);
                                }
                            }
                        }
                        File[] files = f.listFiles();
                        if (files != null) {
                            for (File file : files) {
                                Loader.createLibraryLink(file.getAbsolutePath(), libProperties, null,
                                        linkPaths.toArray(new String[linkPaths.size()]));
                            }
                        }
                    }
                }
                if (paths.length() > 0) {
                    pb.environment().put("BUILD_PATH", paths);
                    pb.environment().put("BUILD_PATH_SEPARATOR", separator);
                }
            }
            int exitValue = pb.inheritIO().start().waitFor();
            if (exitValue != 0) {
                throw new RuntimeException("Process exited with an error: " + exitValue);
            }
            return false;
        }

        if (classScanner.getClasses().isEmpty()) {
            return false;
        }

        List<File> outputFiles = new ArrayList<File>();
        Map<String, LinkedHashSet<Class>> map = new LinkedHashMap<String, LinkedHashSet<Class>>();
        for (Class c : classScanner.getClasses()) {
            if (Loader.getEnclosingClass(c) != c) {
                continue;
            }
            ClassProperties p = Loader.loadProperties(c, properties, false);
            if (!p.isLoaded()) {
                logger.warn("Could not load platform properties for " + c);
                continue;
            }
            try {
                if (Arrays.asList(c.getInterfaces()).contains(BuildEnabled.class)) {
                    ((BuildEnabled)c.newInstance()).init(logger, properties, encoding);
                }
            } catch (ClassCastException | InstantiationException | IllegalAccessException e) {
                // fail silently as if the interface wasn't implemented
            }
            String target = p.getProperty("target");
            if (target != null && !c.getName().equals(target)) {
                File f = parse(classScanner.getClassLoader().getPaths(), c);
                if (f != null) {
                    outputFiles.add(f);
                }
                continue;
            }
            String libraryName = outputName != null ? outputName : p.getProperty("platform.library", "");
            if (libraryName.length() == 0) {
                continue;
            }
            LinkedHashSet<Class> classList = map.get(libraryName);
            if (classList == null) {
                map.put(libraryName, classList = new LinkedHashSet<Class>());
            }
            classList.addAll(p.getEffectiveClasses());
        }
        int count = 0;
        for (String libraryName : map.keySet()) {
            LinkedHashSet<Class> classSet = map.get(libraryName);
            Class[] classArray = classSet.toArray(new Class[classSet.size()]);
            boolean result = generate(classArray, libraryName, count == 0, count == map.size() - 1);
        }


        // reset the load flag to let users load compiled libraries
        System.setProperty("org.bytedeco.javacpp.loadlibraries", "true");
        return true;
    }

    /**
     * Simply prints out to the display the command line usage.
     */
    public static void printHelp() {
        String version = Builder.class.getPackage().getImplementationVersion();
        if (version == null) {
            version = "unknown";
        }
        System.out.println(
            "JavaCPP version " + version + "\n" +
            "Copyright (C) 2011-2017 Samuel Audet <samuel.audet@gmail.com>\n" +
            "Project site: https://github.com/bytedeco/javacpp");
        System.out.println();
        System.out.println("Usage: java -jar javacpp.jar [options] [class or package (suffixed with .* or .**)]");
        System.out.println();
        System.out.println("where options include:");
        System.out.println();
        System.out.println("    -classpath <path>      Load user classes from path");
        System.out.println("    -encoding <name>       Character encoding used for input and output files");
        System.out.println("    -d <directory>         Output all generated files to directory");
        System.out.println("    -o <name>              Output everything in a file named after given name");
        System.out.println("    -nocompile             Do not compile or delete the generated source files");
        System.out.println("    -nodelete              Do not delete generated C++ JNI files after compilation");
        System.out.println("    -header                Generate header file with declarations of callbacks functions");
        System.out.println("    -copylibs              Copy to output directory dependent libraries (link and preload)");
        System.out.println("    -copyresources         Copy to output directory resources listed in properties");
        System.out.println("    -jarprefix <prefix>    Also create a JAR file named \"<prefix>-<platform>.jar\"");
        System.out.println("    -properties <resource> Load all properties from resource");
        System.out.println("    -propertyfile <file>   Load all properties from file");
        System.out.println("    -D<property>=<value>   Set property to value");
        System.out.println("    -Xcompiler <option>    Pass option directly to compiler");
        System.out.println();
    }

    /**
     * The terminal shell interface to the Builder.
     *
     * @param args an array of arguments as described by {@link #printHelp()}
     * @throws Exception
     */
    public static void main(String[] args) throws Exception {
        boolean addedClasses = false;
        Builder builder = new Builder();
        for (int i = 0; i < args.length; i++) {
            if ("-help".equals(args[i]) || "--help".equals(args[i])) {
                printHelp();
                System.exit(0);
            } else if ("-classpath".equals(args[i]) || "-cp".equals(args[i]) || "-lib".equals(args[i])) {
                builder.classPaths(args[++i]);
            } else if ("-encoding".equals(args[i])) {
                builder.encoding(args[++i]);
            } else if ("-d".equals(args[i])) {
                builder.outputDirectory(args[++i]);
            } else if ("-o".equals(args[i])) {
                builder.outputName(args[++i]);
            } else if ("-cpp".equals(args[i]) || "-nocompile".equals(args[i])) {
                builder.compile(false);
            } else if ("-nodelete".equals(args[i])) {
                builder.deleteJniFiles(false);
            } else if ("-header".equals(args[i])) {
                builder.header(true);
            } else if ("-copylibs".equals(args[i])) {
                builder.copyLibs(true);
            } else if ("-copyresources".equals(args[i])) {
                builder.copyResources(true);
            } else if ("-jarprefix".equals(args[i])) {
                builder.jarPrefix(args[++i]);
            } else if ("-properties".equals(args[i])) {
                builder.properties(args[++i]);
            } else if ("-propertyfile".equals(args[i])) {
                builder.propertyFile(args[++i]);
            } else if (args[i].startsWith("-D")) {
                builder.property(args[i]);
            } else if ("-Xcompiler".equals(args[i])) {
                builder.compilerOptions(args[++i]);
            } else if (args[i].startsWith("-")) {
                builder.logger.error("Invalid option \"" + args[i] + "\"");
                printHelp();
                System.exit(1);
            } else {
                System.out.println("adding class " + args[i]);
                builder.classesOrPackages(args[i]);
                addedClasses = true;
            }
        }
        if (!addedClasses) {
            builder.classesOrPackages((String[])null);
        }
        builder.build();
    }
}
