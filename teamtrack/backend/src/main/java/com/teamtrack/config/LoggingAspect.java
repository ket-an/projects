package com.teamtrack.config;

import lombok.extern.slf4j.Slf4j;
import org.aspectj.lang.JoinPoint;
import org.aspectj.lang.ProceedingJoinPoint;
import org.aspectj.lang.annotation.*;
import org.springframework.stereotype.Component;

/**
 * Logging Aspect — demonstrates Spring AOP concepts
 *
 * @Aspect          - Marks this class as an AOP aspect (requires @EnableAspectJAutoProxy
 *                    or spring-boot-starter-aop which enables it automatically)
 * @Component       - Registers as Spring bean so AOP weaving is applied
 *
 * @Around          - Intercepts method call; has control to proceed or block it.
 *                    Used here for execution time logging.
 * @Before          - Runs BEFORE the matched method executes. No ability to block.
 * @AfterReturning  - Runs AFTER method returns successfully.
 * @AfterThrowing   - Runs when method throws an exception.
 * @Pointcut        - Named reusable expression that defines where advice applies.
 *
 * Pointcut expression syntax:
 *   execution(modifiers? returnType declaringType? methodName(params) throws?)
 *   within(package.*)  — all methods inside a package
 */
@Aspect
@Component
@Slf4j
public class LoggingAspect {

    /**
     * @Pointcut - Reusable pointcut: all methods in any service class under com.teamtrack
     */
    @Pointcut("within(com.teamtrack..service..*)")
    public void serviceMethods() {}

    /**
     * @Pointcut - All REST controller methods
     */
    @Pointcut("within(@org.springframework.web.bind.annotation.RestController *)")
    public void controllerMethods() {}

    /**
     * @Around - Wraps service method execution; logs entry, exit and elapsed time.
     *           ProceedingJoinPoint.proceed() is the actual method call.
     */
    @Around("serviceMethods()")
    public Object logServiceExecutionTime(ProceedingJoinPoint joinPoint) throws Throwable {
        String methodName = joinPoint.getSignature().toShortString();
        long start = System.currentTimeMillis();

        log.debug(">>> Service call: {}", methodName);
        try {
            Object result = joinPoint.proceed();
            long elapsed = System.currentTimeMillis() - start;
            log.debug("<<< Service call: {} completed in {}ms", methodName, elapsed);
            return result;
        } catch (Throwable ex) {
            log.warn("!!! Service call: {} threw {}: {}", methodName,
                ex.getClass().getSimpleName(), ex.getMessage());
            throw ex;
        }
    }

    /**
     * @Before - Logs incoming HTTP request to every controller method
     */
    @Before("controllerMethods()")
    public void logControllerRequest(JoinPoint joinPoint) {
        log.debug("HTTP → {}", joinPoint.getSignature().toShortString());
    }

    /**
     * @AfterReturning - Logs when a controller method returns successfully.
     *                   'returning' binds the method's return value to 'result'.
     */
    @AfterReturning(pointcut = "controllerMethods()", returning = "result")
    public void logControllerResponse(JoinPoint joinPoint, Object result) {
        log.debug("HTTP ← {} returned {}", joinPoint.getSignature().getDeclaringTypeName(),
            result != null ? result.getClass().getSimpleName() : "null");
    }

    /**
     * @AfterThrowing - Fires when any service method throws; provides centralised
     *                  exception-level logging (separate from GlobalExceptionHandler).
     */
    @AfterThrowing(pointcut = "serviceMethods()", throwing = "ex")
    public void logServiceException(JoinPoint joinPoint, Throwable ex) {
        log.error("Exception in {}: {}", joinPoint.getSignature().toShortString(),
            ex.getMessage());
    }
}
